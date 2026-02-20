import io
import json
import os
import sys
from collections import defaultdict

import librosa
import soundfile as sf
import torch
import wandb
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import bytes2mel, SAMPLE_RATE
from audio_vqa.eval.metrics import (normalize, simple_lemma, semantic_match,
                                    classify_error, ERROR_CATEGORIES)


class LLMJudge:
    EQUIVALENCE_PROMPT = (
        "You are judging whether two answers to a visual question are semantically equivalent.\n"
        "Consider cases like these as equivalent:\n"
        "- Synonyms (jeans/denim, cell phone/mobile phone, bicycle/bike)\n"
        "- Number formats (30/thirty, 2/two)\n"
        "- Abbreviations vs full names (usa/united states, uk/united kingdom)\n"
        "- Plural vs singular (horses/horse, cats/cat)\n"
        "- Noun vs adjective forms (america/american, usa/american)\n"
        "- Role vs activity (pilots/flying, chef/cooking)\n"
        "- Minor rephrasing with same meaning\n\n"
        "Question: {question}\n"
        "Answer A: {ref}\n"
        "Answer B: {hyp}\n\n"
        "Are these answers semantically equivalent? Reply ONLY 'yes' or 'no'."
    )

    CLASSIFY_PROMPT = (
        "Classify this visual question-answering error into exactly one category.\n\n"
        "Categories:\n"
        "- counting_error: question asks about quantity/count\n"
        "- spatial_reasoning_error: question involves spatial relationships\n"
        "- knowledge_gap: question requires external/world knowledge\n"
        "- text_reading_error: question asks about text/signs in the image\n"
        "- action_recognition_error: question asks about activities/actions\n"
        "- object_identification_error: question asks to identify an object type\n"
        "- morphological_mismatch: predicted answer is a different word form of the correct answer\n"
        "- sentence_instead_of_word: expected a short answer but got a full sentence\n"
        "- other_incorrect: none of the above\n\n"
        "Question: {question}\n"
        "Expected: {ref}\n"
        "Predicted: {hyp}\n\n"
        "Reply with ONLY the category name."
    )

    def __init__(self, model_id="microsoft/Phi-3-mini-4k-instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    def _generate(self, prompt, max_new_tokens=8):
        self._ensure_loaded()
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(
            self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip().lower()

    def is_equivalent(self, ref, hyp, question=""):
        try:
            prompt = self.EQUIVALENCE_PROMPT.format(
                question=question, ref=ref, hyp=hyp)
            return self._generate(prompt).startswith("yes")
        except Exception:
            return False

    def classify_error(self, ref, hyp, question=""):
        try:
            prompt = self.CLASSIFY_PROMPT.format(
                question=question, ref=ref, hyp=hyp)
            response = self._generate(prompt, max_new_tokens=16)
            for cat in ERROR_CATEGORIES:
                if cat in response:
                    return cat
            return "other_incorrect"
        except Exception:
            return "other_incorrect"


def evaluate_vqa(model, processor, eval_samples,
                 split_name="eval", n=50, run_tag="", judge=None):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    exact_count = 0
    soft_count = 0
    semantic_count = 0
    llm_judge_count = 0
    rouge_scores = []
    total = 0
    results = []

    for i, sample in enumerate(eval_samples[:n]):
        ref_answer = sample["answer"].strip().lower()

        audio_bytes = sample.get("audio_bytes")
        if audio_bytes is None:
            audio_info = sample.get("audio", {})
            if isinstance(audio_info, dict) and "array" in audio_info:
                buf = io.BytesIO()
                sf.write(buf, audio_info["array"],
                         audio_info.get("sampling_rate", 16000), format='WAV')
                audio_bytes = buf.getvalue()
            else:
                audio_bytes = audio_info["bytes"]

        user_msgs = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_bytes},
                {"type": "image", "image": sample["image"]},
            ]},
        ]

        text = processor.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True)
        audio_buf = io.BytesIO(audio_bytes)
        wav, _ = librosa.load(audio_buf, sr=SAMPLE_RATE)
        mel = bytes2mel(wav).squeeze(0).cpu()

        batch = processor(
            text=[text],
            images=[sample["image"]],
            audio=[mel],
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to(model.device) if hasattr(v, 'to') else v
                 for k, v in batch.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **batch, max_new_tokens=64, do_sample=False)

        input_len = batch["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        hyp = processor.tokenizer.decode(
            output_ids, skip_special_tokens=True).strip().lower()

        is_exact_match = (hyp == ref_answer)
        is_soft_match = (ref_answer in hyp or hyp in ref_answer)
        is_semantic_match = semantic_match(ref_answer, hyp)
        is_llm_match = False

        if judge and not is_semantic_match:
            question = sample.get("question_text", "")
            is_llm_match = judge.is_equivalent(ref_answer, hyp, question)

        exact_count += int(is_exact_match)
        soft_count += int(is_soft_match)
        semantic_count += int(is_semantic_match)
        llm_judge_count += int(is_llm_match)

        rouge_l = scorer.score(ref_answer, hyp)['rougeL'].fmeasure
        rouge_scores.append(rouge_l)
        total += 1

        results.append({
            "q": sample.get("question_text", ""), "ref": ref_answer, "hyp": hyp,
            "exact": is_exact_match, "soft": is_soft_match,
            "semantic": is_semantic_match, "llm_judge": is_llm_match,
            "rouge_l": rouge_l,
        })

        if i < 5:
            print(f"  [{split_name} {i+1}] ref={ref_answer} hyp={hyp} "
                  f"exact={is_exact_match} sem={is_semantic_match} rouge={rouge_l:.2f}")

    metrics = {
        "exact_match": exact_count / total if total else 0,
        "soft_match": soft_count / total if total else 0,
        "semantic_match": semantic_count / total if total else 0,
        "llm_judge_match": llm_judge_count / total if total else 0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
        "n": total,
    }

    print(f"  {split_name} (n={total}): exact={metrics['exact_match']:.3f} "
          f"sem={metrics['semantic_match']:.3f} rouge={metrics['rouge_l']:.3f}")

    if run_tag:
        eval_dir = os.path.join(os.path.dirname(__file__), "eval")
        os.makedirs(eval_dir, exist_ok=True)

        with open(os.path.join(eval_dir, f"eval_{split_name}_{run_tag}.json"), "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)

        errors = [r for r in results
                  if not r["exact"] and not r["semantic"] and not r["llm_judge"]]
        if errors:
            _write_error_files(errors, split_name, run_tag, eval_dir, judge)

    model.train()
    return metrics


def _classify(err, judge):
    """Classify an error: use deterministic check first, then LLM judge."""
    cat = classify_error(err["ref"], err["hyp"], err["q"])
    if cat is not None:
        return cat
    if judge:
        return judge.classify_error(err["ref"], err["hyp"], err["q"])
    return "other_incorrect"


def _write_error_files(errors, split_name, run_tag, eval_dir, judge=None):
    for err in errors:
        err["category"] = _classify(err, judge)

    error_taxonomy = defaultdict(list)
    for err in errors:
        error_taxonomy[err["category"]].append(err)

    with open(os.path.join(eval_dir, f"error_cases_{split_name}_{run_tag}.txt"), "w") as f:
        for err in errors:
            f.write(f"Q: {err['q']}\n")
            f.write(f"REF: {err['ref']}  HYP: {err['hyp']}\n")
            f.write(f"Category: {err['category']}\n")
            f.write(f"ROUGE-L: {err['rouge_l']:.2f}\n\n")

    sorted_cats = sorted(error_taxonomy.items(), key=lambda x: -len(x[1]))
    with open(os.path.join(eval_dir, f"error_taxonomy_{split_name}_{run_tag}.md"), "w") as f:
        f.write(f"# Error Taxonomy: {split_name} ({run_tag})\n\n")
        f.write(f"Total errors: {len(errors)}\n\n")
        f.write("| Category | Count | % |\n")
        f.write("|----------|------:|--:|\n")
        for cat, cat_errors in sorted_cats:
            pct = len(cat_errors) / len(errors) * 100
            f.write(f"| {cat} | {len(cat_errors)} | {pct:.0f}% |\n")
        f.write("\n## Examples\n\n")
        for cat, cat_errors in sorted_cats[:5]:
            ex = cat_errors[0]
            f.write(f"### {cat}\n")
            f.write(f"- Q: {ex['q']}\n")
            f.write(f"- REF: {ex['ref']} | HYP: {ex['hyp']}\n\n")


class VQAEarlyStoppingCallback:
    def __init__(self, eval_fn, eval_samples, patience=2, eval_every=100,
                 min_delta=0.01):
        self.eval_fn = eval_fn
        self.eval_samples = eval_samples
        self.patience = patience
        self.eval_every = eval_every
        self.min_delta = min_delta
        self.best_acc = 0.0
        self.wait = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every != 0 or state.global_step == 0:
            return
        metrics = self.eval_fn(
            model, self.eval_samples,
            f"val_step{state.global_step}",
            n=min(30, len(self.eval_samples)))
        acc = metrics["semantic_match"]
        wandb.log({
            "val_semantic_match": acc,
            "val_soft_match": metrics["soft_match"],
            "val_exact_match": metrics["exact_match"],
            "val_rouge_l": metrics["rouge_l"],
            "step": state.global_step,
        })
        if acc > self.best_acc + self.min_delta:
            self.best_acc = acc
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            print(f"  Early stop at step {state.global_step} (best={self.best_acc:.3f})")
            control.should_training_stop = True
