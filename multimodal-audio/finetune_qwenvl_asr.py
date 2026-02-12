import os
import gc
import io
import time
import json
import torch
import librosa
import whisper  # needed for pad_or_trim and log_mel_spectrogram
import wandb
from dataclasses import dataclass
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from jiwer import wer
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2VLForConditionalGenerationWithAudio,
    Qwen2VLProcessor,
    Trainer,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class Config:
    # Stage 1: audio projector-only (0.2% params trainable)
    stg1_steps: int = 22500
    stg1_lr: float = 5e-4
    stg1_batch: int = 2
    stg1_grad_accum: int = 2  
    stg1_warmup: float = 0.1

    # Stage 2: QLoRA on the entire model
    stg2_steps: int = 1000
    stg2_lr: float = 2e-5
    stg2_batch: int = 2
    stg2_grad_accum: int = 4  
    stg2_warmup: float = 0.1
    stg2_early_stopping_patience: int = 5

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj")

    lr_schedule: str = "cosine"
    exp_name: str = ""


cfg = Config()

RUN_TAG = datetime.now().strftime("%m%d%H%M")
if cfg.exp_name:
    RUN_TAG = f"{RUN_TAG}_{cfg.exp_name}"

load_dotenv(os.path.expanduser("~/ft/.env"))

HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)

HF_repo_base = "hyan/qwen_speech_base"
HF_repo_stg1 = "hyan/qwen_speech_stage1"
HF_repo_ft   = "hyan/qwen-speech-ft"

api = HfApi()
api.create_repo(HF_repo_stg1, exist_ok=True)
api.create_repo(HF_repo_ft, exist_ok=True)

print(f"Run tag: {RUN_TAG}")
print(f"Config: {cfg}")



WHISPER_N_MELS = 128  


def bytes2mel(audio_np):
    audio = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio, n_mels=WHISPER_N_MELS)
    return mel.unsqueeze(0)


def bytes_to_waveform(audio_bytes):
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=16000)
    return y


def format_data(record):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": record["wav"]["bytes"]},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": record["text"]},
            ],
        },
    ]
    return {"messages": conversation}


def extract_audio_bytes(msgs):
    for msg in msgs:
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "audio":
                return content.get("audio")
    return None


AUDIO_TEMPLATE_ORIG = (
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}"
    "{% endif %}{% endfor %}<|im_end|>"
)

AUDIO_TEMPLATE_REPLACEMENT = (
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}"
    "<|audio_start|><|audio_pad|><|audio_end|>"
    "{% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}"
    "{% endif %}{% endfor %}<|im_end|>"
)


def patch_processor(proc):
    special_tokens = ["<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"]
    proc.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tmpl = proc.tokenizer.chat_template
    if "audio" not in tmpl:
        tmpl = tmpl.replace(AUDIO_TEMPLATE_ORIG, AUDIO_TEMPLATE_REPLACEMENT, 1)
        proc.tokenizer.chat_template = tmpl
    proc.chat_template = proc.tokenizer.chat_template
    assert "audio" in proc.chat_template, "Chat template patch failed!"
    return proc


processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = patch_processor(processor)
processor.save_pretrained("./qwen2-vl-speech-processor")


def prepare_audio_encoder(model):
    """Upcast whisper encoder to fp32 and freeze it.
    """
    model.audio_encoder.float()
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    model.audio_encoder.eval()
    device = next(model.audio_encoder.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(1, WHISPER_N_MELS, 3000, device=device)
        test_out = model.audio_encoder(dummy)
        assert not torch.isnan(test_out).any(), "Whisper encoder produces NaN — fp32 upcast insufficient, may need fresh weights"
    return model


def collate_fn(examples):
    texts = []
    mel_list = []
    raw_prompt_lens = []

    for ex in examples:
        formatted = format_data(ex)
        msgs = formatted["messages"]
        full_text = processor.apply_chat_template(msgs, tokenize=False)

        prompt_msgs = [msgs[0]]
        prompt_text = processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        raw_prompt_ids = processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        raw_prompt_len = raw_prompt_ids.shape[1]

        audio_bytes = extract_audio_bytes(msgs)
        if audio_bytes is None:
            continue
        wav = bytes_to_waveform(audio_bytes)
        mel = bytes2mel(wav).squeeze(0).cpu()
        texts.append(full_text)
        mel_list.append(mel)
        raw_prompt_lens.append(raw_prompt_len)

    if len(texts) == 0:
        return None

    batch = processor(text=texts, audio=mel_list, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for audio_token_id in [151657, 151658, 151659]:
        labels[labels == audio_token_id] = -100
    AUDIO_PAD_ID = 151658
    for i, raw_prompt_len in enumerate(raw_prompt_lens):
        audio_pad_count = (batch["input_ids"][i] == AUDIO_PAD_ID).sum().item()
        if audio_pad_count > 0:
            expanded_prompt_len = raw_prompt_len + audio_pad_count - 1
        else:
            expanded_prompt_len = raw_prompt_len
        labels[i, :expanded_prompt_len] = -100

    batch["labels"] = labels
    return batch


def evaluate_wer(model, processor, eval_samples, split_name="eval"):
    model.eval()
    references = []
    hypotheses = []

    for i, sample in enumerate(eval_samples):
        ref_text = sample["text"].strip().lower()
        references.append(ref_text)

        user_msgs = [
            {"role": "user", "content": [
                {"type": "audio", "audio": sample["wav"]["bytes"]},
                {"type": "text", "text": "Transcribe this audio."},
            ]},
        ]
        text = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
        wav = bytes_to_waveform(sample["wav"]["bytes"])
        mel = bytes2mel(wav).squeeze(0).cpu()

        batch = processor(text=[text], audio=[mel], padding=True, return_tensors="pt")
        batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}

        with torch.no_grad():
            generated_ids = model.generate(**batch, max_new_tokens=256, do_sample=False)

        input_len = batch["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        hyp_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
        hypotheses.append(hyp_text)

        print(f"  [{split_name} {i+1}] REF: {ref_text[:100]}")
        print(f"  [{split_name} {i+1}] HYP: {hyp_text[:100]}")

    error_rate = wer(references, hypotheses)
    print(f"\n  {split_name} WER: {error_rate:.4f} ({error_rate*100:.1f}%)")
    model.train()
    return error_rate



train_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["small/train-0000*", "small/train-0001*"],
    num_proc=12,
)
test_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["test/test-00000*"],
    num_proc=12,
)
train_dataset = train_dataset["train"]
test_dataset = test_dataset["train"]
test_dataset = test_dataset.select(range(100))
print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

eval_train_samples = [train_dataset[i] for i in range(10)]
eval_val_samples = [test_dataset[i] for i in range(10)]


def clear_memory():
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)


# ==============================================================
# STAGE 1: Train only audio projector (freeze LM + whisper)
# ==============================================================

model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
    HF_repo_base, torch_dtype=torch.bfloat16, device_map="auto"
)
model = prepare_audio_encoder(model)

processor = patch_processor(Qwen2VLProcessor.from_pretrained("./qwen2-vl-speech-processor"))

for name, param in model.named_parameters():
    if "audio_projector" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

wandb.login(key=os.environ["wandb"])
wandb.init(project="qwen2vl_speech", name=f"speech_stg1_{RUN_TAG}")

sft_config_stg1 = SFTConfig(
    output_dir=f"./speech_stg1_{RUN_TAG}",
    max_steps=cfg.stg1_steps,
    per_device_train_batch_size=cfg.stg1_batch,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=cfg.stg1_grad_accum,
    learning_rate=cfg.stg1_lr,
    warmup_ratio=cfg.stg1_warmup,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    report_to="wandb",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)

trainer_stage1 = SFTTrainer(
    model=model,
    args=sft_config_stg1,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    processing_class=processor,
)

print(f"  Stage 1: {cfg.stg1_steps} steps, LR={cfg.stg1_lr}, eff_batch={cfg.stg1_batch * cfg.stg1_grad_accum}")
trainer_stage1.train()
print("  Stage 1 training complete!")

wer_train_stg1 = evaluate_wer(model, processor, eval_train_samples, "stg1_train")
wer_val_stg1 = evaluate_wer(model, processor, eval_val_samples, "stg1_val")
wandb.log({"stg1_wer_train": wer_train_stg1, "stg1_wer_val": wer_val_stg1})

os.makedirs("./models", exist_ok=True)
stg1_local = f"./models/stg1_{RUN_TAG}"
model.save_pretrained(stg1_local)
processor.save_pretrained(stg1_local)
model.push_to_hub(HF_repo_stg1)
processor.push_to_hub(HF_repo_stg1)
print(f"  Stage 1 saved to {stg1_local}")
wandb.finish()

del model, trainer_stage1
clear_memory()

# ==============================================================
# STAGE 2: LoRA fine-tuning of LM (whisper stays frozen)
# ==============================================================

model2 = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
    stg1_local, torch_dtype=torch.bfloat16, device_map="auto"
)
model2 = prepare_audio_encoder(model2)

processor = patch_processor(Qwen2VLProcessor.from_pretrained(stg1_local))

lora_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    target_modules=list(cfg.lora_target_modules),
    bias="none",
    task_type="CAUSAL_LM",
)
model2 = get_peft_model(model2, lora_config)
model2.print_trainable_parameters()

wandb.init(project="qwen2vl_speech", name=f"speech_stg2_{RUN_TAG}")

training_args = SFTConfig(
    output_dir=f"./speech_stg2_{RUN_TAG}",
    max_steps=cfg.stg2_steps,
    per_device_train_batch_size=cfg.stg2_batch,
    gradient_accumulation_steps=cfg.stg2_grad_accum,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=2,
    save_strategy="steps",
    save_steps=250,
    learning_rate=cfg.stg2_lr,
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=cfg.stg2_warmup,
    lr_scheduler_type=cfg.lr_schedule,
    report_to="wandb",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model2,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.stg2_early_stopping_patience)],
)

print(f"  Stage 2: {cfg.stg2_steps} steps, LR={cfg.stg2_lr}, LoRA r={cfg.lora_r}, early_stop={cfg.stg2_early_stopping_patience}")
trainer.train()
print("  Stage 2 training complete!")

wer_train_stg2 = evaluate_wer(model2, processor, eval_train_samples, "stg2_train")
wer_val_stg2 = evaluate_wer(model2, processor, eval_val_samples, "stg2_val")
wandb.log({"stg2_wer_train": wer_train_stg2, "stg2_wer_val": wer_val_stg2})

merged_model = model2.merge_and_unload()

stg2_local = f"./models/stg2_{RUN_TAG}"
merged_model.save_pretrained(stg2_local)
processor.save_pretrained(stg2_local)
merged_model.push_to_hub(HF_repo_ft)
processor.push_to_hub(HF_repo_ft)
print(f"  Stage 2 saved to {stg2_local}")
wandb.finish()



results = {
    "run_tag": RUN_TAG,
    "exp_name": cfg.exp_name,
    "stg1_steps": cfg.stg1_steps, "stg1_lr": cfg.stg1_lr,
    "stg2_steps": cfg.stg2_steps, "stg2_lr": cfg.stg2_lr,
    "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
    "stg1_wer_train": wer_train_stg1, "stg1_wer_val": wer_val_stg1,
    "stg2_wer_train": wer_train_stg2, "stg2_wer_val": wer_val_stg2,
}

with open(f"results_{RUN_TAG}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"RESULTS_JSON {json.dumps(results)}")
