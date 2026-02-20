"""Finetune Qwen2-VL 7B for Speech Understanding (ASR).

Training loop:
  Stage 1: Train audio projector only (optional LR sweep first)
  Stage 2: Full fine-tune or QLoRA (configurable via YAML)

Usage:
    python finetune_qwenvl_asr.py --config configs/config1.yml
"""

import argparse
import io
import json
import os
import sys
import torch
import yaml
import wandb
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from jiwer import wer as compute_wer
from transformers import (
    Qwen2VLForConditionalGenerationWithAudio,
    Qwen2VLProcessor,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import (bytes2mel, bytes_to_waveform, extract_audio_bytes,
                  patch_processor, clear_memory,
                  WHISPER_N_MELS, SAMPLE_RATE)


def chunk_waveform(audio_np, chunk_sec):
    if chunk_sec <= 0 or len(audio_np) <= chunk_sec * SAMPLE_RATE:
        return [audio_np]
    chunk_len = chunk_sec * SAMPLE_RATE
    chunks = []
    for start in range(0, len(audio_np), chunk_len):
        chunks.append(audio_np[start:start + chunk_len])
    return chunks


def format_data(record):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "audio", "audio": record["wav"]["bytes"]},
            {"type": "text", "text": "Transcribe this audio."},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": record["text"]},
        ]},
    ]}


def make_collate_fn(processor):
    def collate_fn(examples):
        texts, mel_list, raw_lens = [], [], []
        for ex in examples:
            fmt = format_data(ex)
            msgs = fmt["messages"]
            full_text = processor.apply_chat_template(msgs, tokenize=False)
            prompt_text = processor.apply_chat_template(
                [msgs[0]], tokenize=False, add_generation_prompt=True)
            raw_len = processor.tokenizer(
                prompt_text, return_tensors="pt")["input_ids"].shape[1]
            audio_bytes = extract_audio_bytes(msgs)
            if audio_bytes is None:
                continue
            wav = bytes_to_waveform(audio_bytes)
            mel = bytes2mel(wav).squeeze(0).cpu()
            texts.append(full_text)
            mel_list.append(mel)
            raw_lens.append(raw_len)
        if not texts:
            return None
        batch = processor(
            text=texts, audio=mel_list, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tid in [151657, 151658, 151659]:
            labels[labels == tid] = -100
        audio_pad_id = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        for i, rl in enumerate(raw_lens):
            n_audio = (batch["input_ids"][i] == audio_pad_id).sum().item()
            pl = rl + max(0, n_audio - 1)
            labels[i, :pl] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def transcribe_single(model, processor, mel):
    user_msgs = [{"role": "user", "content": [
        {"type": "audio", "audio": b"placeholder"},
        {"type": "text", "text": "Transcribe this audio."},
    ]}]
    text = processor.apply_chat_template(
        user_msgs, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[text], audio=[mel], padding=True, return_tensors="pt")
    batch = {k: v.to(model.device) if hasattr(v, "to") else v
             for k, v in batch.items()}
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=256, do_sample=False)
    return processor.tokenizer.decode(
        gen[0][batch["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()


def evaluate_wer(model, processor, eval_samples, split_name="eval",
                 chunk_sec=0):
    model.eval()
    refs, hyps = [], []
    for i, sample in enumerate(eval_samples):
        ref = sample["text"].strip().lower()
        refs.append(ref)
        wav = bytes_to_waveform(sample["wav"]["bytes"])
        chunks = chunk_waveform(wav, chunk_sec)
        parts = []
        for chunk in chunks:
            mel = bytes2mel(chunk).squeeze(0).cpu()
            parts.append(transcribe_single(model, processor, mel))
        hyp = " ".join(parts).lower()
        hyps.append(hyp)
    err = compute_wer(refs, hyps)
    print(f"  {split_name} WER: {err:.4f} ({err*100:.1f}%) on {len(eval_samples)} samples")
    model.train()
    return err


class WERCallback(TrainerCallback):
    def __init__(self, model, processor, eval_samples, eval_steps, patience,
                 run_tag=""):
        self.model = model
        self.processor = processor
        self.eval_samples = eval_samples
        self.eval_steps = eval_steps
        self.patience = patience
        self.run_tag = run_tag
        self.best_wer = float("inf")
        self.best_step = 0
        self.wait = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.eval_steps != 0 or step == 0:
            return
        wer = evaluate_wer(self.model, self.processor, self.eval_samples,
                           f"step_{step}")
        wandb.log({"wer_val": wer, "step": step})

        if wer < self.best_wer:
            self.best_wer, self.best_step, self.wait = wer, step, 0
            save_dir = f"./models/stg2_best_{self.run_tag}"
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_pretrained(save_dir)
            self.processor.save_pretrained(save_dir)
        else:
            self.wait += 1

        if self.wait >= self.patience:
            print(f"Early stopping at step {step}: best WER={self.best_wer:.4f}")
            control.should_training_stop = True


def run_sweep(cfg, hf_base, wandb_project, run_tag, processor,
              train_dataset, test_dataset, collate):
    sweep_cfg = cfg["sweep"]
    sweep_steps = sweep_cfg["steps"]
    sweep_configs = sweep_cfg["configs"]
    stg1_batch = cfg["stg1"]["batch_size"]

    sweep_results = []

    for ci, scfg in enumerate(sweep_configs):
        model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
            hf_base, torch_dtype=torch.bfloat16, device_map="auto")
        proc = patch_processor(
            Qwen2VLProcessor.from_pretrained("./qwen2-vl-speech-processor"))
        model.reload_audio_encoder()
        for n, p in model.named_parameters():
            p.requires_grad = "audio_projector" in n

        wandb.init(project=wandb_project,
                   name=f"sweep_{scfg['name']}_{run_tag}",
                   group=f"sweep_{run_tag}", config=scfg, reinit=True)

        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=f"./sweep_{scfg['name']}_{run_tag}",
                max_steps=sweep_steps,
                per_device_train_batch_size=stg1_batch,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=scfg["grad_accum"],
                learning_rate=scfg["lr"], warmup_ratio=scfg["warmup"],
                logging_steps=10, eval_strategy="steps", eval_steps=50,
                bf16=True, report_to="wandb",
                dataset_kwargs={"skip_prepare_dataset": True},
                remove_unused_columns=False),
            train_dataset=train_dataset, eval_dataset=test_dataset,
            data_collator=collate, processing_class=proc)

        trainer.train()
        log = trainer.state.log_history
        train_losses = [e["loss"] for e in log if "loss" in e]
        eval_losses = [e["eval_loss"] for e in log if "eval_loss" in e]
        final_train = train_losses[-1] if train_losses else float("inf")
        final_eval = eval_losses[-1] if eval_losses else float("inf")
        sweep_results.append({**scfg, "train_loss": final_train, "eval_loss": final_eval})
        wandb.log({"final_train_loss": final_train, "final_eval_loss": final_eval})
        wandb.finish()
        print(f"    => train={final_train:.4f}, eval={final_eval:.4f}")
        del model, trainer
        clear_memory()

    best = min(sweep_results, key=lambda x: x["eval_loss"])
    print(f"\nBEST: {best['name']} (eval={best['eval_loss']:.4f})")
    return best["lr"], best["grad_accum"], best["warmup"]


def train_stage1(cfg, best_lr, best_ga, best_warmup, hf_base, hf_stg1,
                 wandb_project, run_tag, processor,
                 train_dataset, test_dataset, collate,
                 eval_train_samples, eval_val_samples):
    stg1 = cfg["stg1"]

    model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
        hf_base, torch_dtype=torch.bfloat16, device_map="auto")
    proc = patch_processor(
        Qwen2VLProcessor.from_pretrained("./qwen2-vl-speech-processor"))
    model.reload_audio_encoder()

    for n, p in model.named_parameters():
        p.requires_grad = "audio_projector" in n

    wandb.init(project=wandb_project, name=f"stg1_{run_tag}",
               group=f"stg1_{run_tag}",
               config={"lr": best_lr, "grad_accum": best_ga,
                       "warmup": best_warmup, "steps": stg1["steps"]})

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=f"./stg1_{run_tag}", max_steps=stg1["steps"],
            per_device_train_batch_size=stg1["batch_size"],
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=best_ga,
            learning_rate=best_lr, warmup_ratio=best_warmup,
            logging_steps=10, eval_strategy="steps", eval_steps=100,
            save_strategy="steps", save_steps=500,
            bf16=True, report_to="wandb",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False),
        train_dataset=train_dataset, eval_dataset=test_dataset,
        data_collator=collate, processing_class=proc)

    trainer.train()

    wer_t = evaluate_wer(model, proc, eval_train_samples, "stg1_train")
    wer_v = evaluate_wer(model, proc, eval_val_samples[:10], "stg1_val")
    wandb.log({"stg1_wer_train": wer_t, "stg1_wer_val": wer_v})

    stg1_local = f"./models/stg1_{run_tag}"
    os.makedirs(stg1_local, exist_ok=True)
    model.save_pretrained(stg1_local)
    proc.save_pretrained(stg1_local)
    model.push_to_hub(hf_stg1)
    proc.push_to_hub(hf_stg1)
    wandb.finish()

    del model, trainer
    clear_memory()
    return stg1_local, wer_t, wer_v


def train_stage2(cfg, stg1_local, hf_ft, wandb_project, run_tag,
                 train_dataset, test_dataset, collate,
                 eval_train_samples, eval_val_samples):
    stg2 = cfg["stg2"]
    use_qlora = cfg.get("qlora", True)

    if use_qlora:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        quant_cfg = cfg.get("quantization", {})
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quant_cfg.get(
                "bnb_4bit_use_double_quant", True),
            llm_int8_skip_modules=["audio_encoder", "audio_projector"],
        )
        model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
            stg1_local, quantization_config=bnb_config, device_map="auto")
    else:
        model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
            stg1_local, torch_dtype=torch.bfloat16).to("cuda")

    processor = patch_processor(
        Qwen2VLProcessor.from_pretrained(stg1_local))
    model.reload_audio_encoder()

    if use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

        lora_cfg = cfg.get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules",
                                        ["q_proj", "v_proj", "k_proj"]),
            bias="none",
            task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if "audio_projector" in n:
                p.requires_grad = True

        model.print_trainable_parameters()
        optim = "paged_adamw_8bit"
    else:
        for n, p in model.named_parameters():
            p.requires_grad = True
        optim = "adamw_torch"

    wandb.init(project=wandb_project, name=f"stg2_{run_tag}",
               group=f"stg2_{run_tag}",
               config={"lr": stg2["lr"], "qlora": use_qlora,
                       "steps": stg2["steps"]})

    callbacks = []
    wer_callback = None
    if use_qlora:
        wer_eval_steps = stg2.get("wer_eval_steps", 50)
        wer_eval_n = stg2.get("wer_eval_samples", 50)
        wer_callback = WERCallback(
            model=model, processor=processor,
            eval_samples=eval_val_samples[:wer_eval_n],
            eval_steps=wer_eval_steps,
            patience=stg2["early_stopping_patience"],
            run_tag=run_tag)
        callbacks.append(wer_callback)
        eval_steps = wer_eval_steps
        save_steps = wer_eval_steps
        load_best = False
    else:
        early_threshold = stg2.get("early_stopping_threshold", 0.0)
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=stg2["early_stopping_patience"],
            early_stopping_threshold=early_threshold))
        eval_steps = 50
        save_steps = 50
        load_best = True

    # QLoRA dequantizes LM embeddings to fp32, but bf16 autocast makes
    # audio_projector output bf16 -- masked_scatter needs matching dtypes.
    if use_qlora:
        def _cast_projector_output(module, input, output):
            return output.float()
        model.audio_projector.register_forward_hook(_cast_projector_output)

    trainer = Trainer(
        model=model,
        args=SFTConfig(
            output_dir=f"./stg2_{run_tag}",
            max_steps=stg2["steps"],
            per_device_train_batch_size=stg2["batch_size"],
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=stg2["grad_accum"],
            gradient_checkpointing=True,
            optim=optim,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=load_best,
            metric_for_best_model="eval_loss" if load_best else None,
            greater_is_better=False if load_best else None,
            learning_rate=stg2["lr"],
            bf16=True,
            max_grad_norm=1.0,
            warmup_ratio=stg2["warmup"],
            lr_scheduler_type=cfg.get("lr_schedule", "cosine"),
            report_to="wandb",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate,
        callbacks=callbacks)

    trainer.train()

    if use_qlora:
        eval_model = model.merge_and_unload()
    else:
        eval_model = model

    wer_t = evaluate_wer(eval_model, processor, eval_train_samples, "stg2_train")
    wer_v = evaluate_wer(eval_model, processor, eval_val_samples[:10], "stg2_val")
    wandb.log({"stg2_wer_train": wer_t, "stg2_wer_val": wer_v})

    stg2_local = f"./models/stg2_{run_tag}"
    os.makedirs(stg2_local, exist_ok=True)
    eval_model.save_pretrained(stg2_local)
    processor.save_pretrained(stg2_local)
    eval_model.push_to_hub(hf_ft)
    processor.push_to_hub(hf_ft)
    wandb.finish()

    extra = {}
    if wer_callback:
        extra["stg2_best_wer"] = wer_callback.best_wer
        extra["stg2_best_step"] = wer_callback.best_step
    return wer_t, wer_v, extra


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen2-VL for ASR")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_tag = datetime.now().strftime("%m%d%H%M")
    exp_name = cfg.get("exp_name", "")
    if exp_name:
        run_tag = f"{run_tag}_{exp_name}"

    load_dotenv(os.path.expanduser("~/ft/.env"))
    login(token=os.environ["HF_TOKEN"])

    hf_base = cfg["hf_repo_base"]
    hf_stg1 = cfg["hf_repo_stg1"]
    hf_ft = cfg["hf_repo_ft"]

    api = HfApi()
    api.create_repo(hf_stg1, exist_ok=True)
    api.create_repo(hf_ft, exist_ok=True)

    wandb_project = cfg.get("wandb_project", "qwen2vl_speech")
    wandb.login(key=os.environ["wandb"])

    processor = patch_processor(
        Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"))
    processor.save_pretrained("./qwen2-vl-speech-processor")

    train_dataset = load_dataset(
        "speechbrain/LargeScaleASR",
        data_files=["small/train-0000*", "small/train-0001*"],
        num_proc=12)["train"]
    test_dataset = load_dataset(
        "speechbrain/LargeScaleASR",
        data_files=["test/test-00000*"],
        num_proc=12)["train"].select(range(200))

    eval_train_samples = [train_dataset[i] for i in range(10)]
    eval_val_samples = [test_dataset[i] for i in range(
        cfg["stg2"].get("wer_eval_samples", 50))]

    collate = make_collate_fn(processor)

    stg1 = cfg["stg1"]
    best_lr, best_ga, best_warmup = stg1["lr"], stg1["grad_accum"], stg1["warmup"]

    sweep_cfg = cfg.get("sweep")
    if sweep_cfg and sweep_cfg.get("configs"):
        best_lr, best_ga, best_warmup = run_sweep(
            cfg, hf_base, wandb_project, run_tag, processor,
            train_dataset, test_dataset, collate)

    stg1_local, wer_t1, wer_v1 = train_stage1(
        cfg, best_lr, best_ga, best_warmup,
        hf_base, hf_stg1, wandb_project, run_tag, processor,
        train_dataset, test_dataset, collate,
        eval_train_samples, eval_val_samples)
    print(f"Stage 1: train WER={wer_t1:.3f}, val WER={wer_v1:.3f}")

    wer_t2, wer_v2, extra = train_stage2(
        cfg, stg1_local, hf_ft, wandb_project, run_tag,
        train_dataset, test_dataset, collate,
        eval_train_samples, eval_val_samples)
    print(f"Stage 2: train WER={wer_t2:.3f}, val WER={wer_v2:.3f}")


    results = {
        "run_tag": run_tag, "exp_name": exp_name,
        "qlora": cfg.get("qlora", True),
        "stg1_wer_train": wer_t1, "stg1_wer_val": wer_v1,
        "stg2_wer_train": wer_t2, "stg2_wer_val": wer_v2,
        **extra,
    }
    with open(f"results_{run_tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"RESULTS_JSON {json.dumps(results)}")


if __name__ == "__main__":
    main()
