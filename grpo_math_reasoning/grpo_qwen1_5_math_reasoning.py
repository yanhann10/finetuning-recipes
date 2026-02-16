import gc
import os
import re
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

load_dotenv(".env")
login(token=os.environ["hf"])
wandb.login(key=os.environ["wandb"])
wandb.init(project="qwen2vl_reason_narrow_range", name="reason_v3_run_2")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SYSTEM_PROMPT = """
You are a mathematical reasoning engine. Solve the problem step by step.

Put your reasoning inside <thinking>...</thinking>.
Put ONLY the final integer answer inside <answer>...</answer>.

Example:
<thinking>
2 + 3 = 5
5 * 4 = 20
</thinking>
<answer>
20
</answer>
"""

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_REPO = "hyan/reason_v3_run_2"



ds = load_dataset("openai/gsm8k", "main")

def format_data(x):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_answer(x["answer"]),
        "question": x["question"],
    }

train_ds = ds["train"].map(format_data)
test_ds = ds["test"].map(format_data)


def extract_answer(text):
    #in this enhanced version, delineate answer correctness with formatting correctness. if answer not in tag, extract last number from the text..
    if text is None:
        return None
    text = text.replace(",", "").replace("$", "")
    tag_match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
    
    if tag_match:
        nums = re.findall(r"-?\d+", tag_match.group(1))
        if nums: 
            return nums[-1]

    all_nums = re.findall(r"-?\d+", text)
    return all_nums[-1] if all_nums else None

def format_reward(completions, **kwargs) -> list[float]:
    #tiny reward for having format tags
    texts = [c[0]["content"] for c in completions]
    return [
        0.1 if re.search(r"<thinking>.*?</thinking>\s*<answer>.*?</answer>", t, re.DOTALL)
        else 0.0
        for t in texts
    ]


def int_reward(completions, **kwargs) -> list[float]:
    #tiny reward for having integer answer
    texts = [c[0]["content"] for c in completions]
    rewards = []
    for t in texts:
        m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", t)
        if m:
            ans = m.group(1).strip().replace(",", "").replace("$", "")
            rewards.append(0.1 if re.fullmatch(r"-?\d+", ans) else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def correct_reward(completions, answer, **kwargs) -> list[float]:
    #impt to keep correctness the main reward
    texts = [c[0]["content"] for c in completions]
    gts = answer if isinstance(answer, list) else [answer] * len(texts)
    rewards = []
    for t, gt in zip(texts, gts):
        pred = extract_answer(t)
        if pred is None or gt is None:
            rewards.append(0.0)
            continue
        p = re.sub(r"[^\d\-]", "", pred)
        g = re.sub(r"[^\d\-]", "", str(gt))
        rewards.append(1.0 if (p == g and len(p) > 0) else 0.0)
    return rewards


tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)


@torch.no_grad()
def evaluate(mdl, tok, dataset, n=50, max_tokens=300):
    mdl.eval()
    correct = total = format_ok = 0
    samples = dataset.select(range(min(n, len(dataset))))

    for s in tqdm(samples, desc="Evaluating"):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["question"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(mdl.device)
        out = mdl.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

        pred = extract_answer(resp)
        gt = s["answer"]
        is_correct = False
        if pred is not None and gt is not None:
            p = re.sub(r"[^\d\-]", "", pred)
            g = re.sub(r"[^\d\-]", "", str(gt))
            is_correct = (p == g and len(p) > 0)

        has_format = bool(re.search(r"<thinking>.*?</thinking>\s*<answer>.*?</answer>", resp, re.DOTALL))
        correct += is_correct
        format_ok += has_format
        total += 1

    acc = correct / max(total, 1)
    format_rate = format_ok / max(total, 1)
    print(f"\nEval: acc={acc:.3f} ({correct}/{total}), format={format_rate:.3f}")
    return {"accuracy": acc, "format_rate": format_rate, "correct": correct, "total": total}


baseline = evaluate(model, tok, test_ds, n=50)
wandb.log({"eval/baseline_accuracy": baseline["accuracy"], "eval/baseline_format": baseline["format_rate"]})

grpo_args = GRPOConfig(
    output_dir="./grpo_v3_run_2",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=200,
    num_generations=8,
    beta=0.01,
    max_prompt_length=256,
    max_completion_length=450,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=0.1,
    weight_decay=0.1,
    report_to="wandb",
    run_name="reason_v3_run_2",
    remove_unused_columns=False,
    temperature=1.0,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    reward_funcs=[format_reward, int_reward, correct_reward],
    peft_config=lora_cfg,
    processing_class=tok,
)

trainer.train()
trainer.save_model("./grpo_v3_run_2/final")

del model
gc.collect()
torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
finetuned = PeftModel.from_pretrained(base, "./grpo_v3_run_2/final", torch_dtype=torch.bfloat16)
finetuned.eval()

post = evaluate(finetuned, tok, test_ds, n=100)
wandb.log({"eval/post_accuracy": post["accuracy"], "eval/post_format": post["format_rate"]})

print(f"\nBASELINE: {baseline['accuracy']*100:.1f}%")
print(f"POST:     {post['accuracy']*100:.1f}% ({post['correct']}/{post['total']})")

merged = finetuned.merge_and_unload()
merged.push_to_hub(HF_REPO, token=os.environ["hf"])
tok.push_to_hub(HF_REPO, token=os.environ["hf"])
merged.save_pretrained("./grpo_v3_run_2/merged")
tok.save_pretrained("./grpo_v3_run_2/merged")

wandb.finish()
