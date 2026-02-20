"""Fine-tune Qwen2-VL+Audio for Audio VQA.

Input: audio question + image
Output: text answer.

Training Data: hyan/qwen-audio-vqa-ft based on a-okvqa (https://github.com/allenai/aokvqa) with text input tts-ed into audio.
"""

import argparse
import io
import json
import os
import sys
import time

import librosa
import soundfile as sf
import torch
import wandb
import yaml
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2VLForConditionalGenerationWithAudio,
    Qwen2VLProcessor,
    Trainer,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import (bytes2mel, bytes_to_waveform, extract_audio_bytes,
                  extract_image, patch_processor, clear_memory,
                  AUDIO_PAD_ID, IMAGE_PAD_ID, SAMPLE_RATE)
from audio_vqa.eval_vqa import evaluate_vqa, VQAEarlyStoppingCallback


def format_data(record):
    audio_bytes = record.get("audio_bytes")
    if audio_bytes is None:
        audio_info = record.get("audio", {})
        if isinstance(audio_info, dict) and "bytes" in audio_info:
            audio_bytes = audio_info["bytes"]
        elif isinstance(audio_info, dict) and "array" in audio_info:
            buf = io.BytesIO()
            sf.write(buf, audio_info["array"],
                     audio_info.get("sampling_rate", 16000), format='WAV')
            audio_bytes = buf.getvalue()
        else:
            raise ValueError(f"Unexpected audio format: {type(audio_info)}")

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": ("Listen to the audio question and look at the image. "
                          "Answer with only the most relevant word or short phrase "
                          "(1-3 words). Do not explain your reasoning.")},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_bytes},
                {"type": "image", "image": record["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": record["answer"]},
            ],
        },
    ]
    return {"messages": conversation}


def make_collate_fn(processor):
    def collate_fn(examples):
        texts = []
        mel_list = []
        image_list = []
        raw_prompt_lens = []

        for ex in examples:
            formatted = format_data(ex)
            msgs = formatted["messages"]
            full_text = processor.apply_chat_template(msgs, tokenize=False)

            prompt_msgs = [msgs[0]]
            prompt_text = processor.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True)
            raw_prompt_ids = processor.tokenizer(
                prompt_text, return_tensors="pt")["input_ids"]
            raw_prompt_len = raw_prompt_ids.shape[1]

            audio_bytes = extract_audio_bytes(msgs)
            if audio_bytes is None:
                continue

            audio_buf = io.BytesIO(audio_bytes)
            wav, _ = librosa.load(audio_buf, sr=SAMPLE_RATE)
            mel = bytes2mel(wav).squeeze(0).cpu()

            image = extract_image(msgs)
            if image is None:
                continue

            texts.append(full_text)
            mel_list.append(mel)
            image_list.append(image)
            raw_prompt_lens.append(raw_prompt_len)

        if not texts:
            return None

        batch = processor(
            text=texts,
            images=image_list,
            audio=mel_list,
            padding=True,
            return_tensors="pt",
        )

        seq_len = batch["input_ids"].shape[1]
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        for special_id in [151657, 151658, 151659]:
            labels[labels == special_id] = -100

        for i, raw_plen in enumerate(raw_prompt_lens):
            audio_pad_count = (batch["input_ids"][i] == AUDIO_PAD_ID).sum().item()
            image_pad_count = (batch["input_ids"][i] == IMAGE_PAD_ID).sum().item()

            expanded_prompt_len = raw_plen
            if audio_pad_count > 0:
                expanded_prompt_len += audio_pad_count - 1
            if image_pad_count > 0:
                expanded_prompt_len += image_pad_count - 1

            labels[i, :expanded_prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def train_stage1(cfg, processor, train_dataset, val_dataset,
                 eval_train_samples, eval_val_samples, collate, run_tag):
    stg1 = cfg["stg1"]
    base_model = cfg["base_model"]
    hf_repo_stg1 = cfg["hf_repo_stg1"]
    wandb_project = cfg["wandb_project"]

    print("\nStage 1: Training audio projector only")
    model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.reload_audio_encoder()

    processor_local = patch_processor(
        Qwen2VLProcessor.from_pretrained("./qwen2-vl-audio-vqa-processor"))

    for name, param in model.named_parameters():
        param.requires_grad = "audio_projector" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    wandb.init(project=wandb_project, name=f"avqa_stg1_{run_tag}")

    output_dir = f"./avqa_stg1_{run_tag}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=stg1["steps"],
        per_device_train_batch_size=stg1["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=stg1["grad_accum"],
        learning_rate=stg1["lr"],
        warmup_ratio=stg1["warmup"],
        logging_steps=10,
        eval_steps=50,
        bf16=True,
        report_to="wandb",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    def _eval_fn(mdl, samples, split, n=50):
        return evaluate_vqa(
            mdl, processor_local, samples,
            split_name=split, n=n, run_tag=run_tag)

    early_stop_cb = VQAEarlyStoppingCallback(
        eval_fn=_eval_fn, eval_samples=eval_val_samples,
        patience=3, eval_every=100)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate,
        processing_class=processor_local,
        callbacks=[early_stop_cb],
    )

    trainer.train()
    print("  Stage 1 complete")

    m_train = _eval_fn(model, eval_train_samples, "train_stg1")
    m_val = _eval_fn(model, eval_val_samples, "val_stg1")
    wandb.log({f"stg1_{k}_train": v for k, v in m_train.items()
               if isinstance(v, (int, float))})
    wandb.log({f"stg1_{k}_val": v for k, v in m_val.items()
               if isinstance(v, (int, float))})

    stg1_local = "./qwen-audio-vqa-stage1"
    model.save_pretrained(stg1_local)
    processor_local.save_pretrained(stg1_local)
    model.push_to_hub(hf_repo_stg1)
    processor_local.push_to_hub(hf_repo_stg1)
    print(f"  Saved to {hf_repo_stg1}")
    wandb.finish()

    del model, trainer
    clear_memory()
    return stg1_local, m_train, m_val


def train_stage2(cfg, stg1_local, train_dataset,
                 eval_train_samples, eval_val_samples, collate, run_tag):
    stg2 = cfg["stg2"]
    hf_repo_ft = cfg["hf_repo_ft"]
    wandb_project = cfg["wandb_project"]
    lr_schedule = cfg.get("lr_schedule", "cosine")

    print("\nStage 2: Full fine-tuning")
    model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
        stg1_local, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    model.reload_audio_encoder()

    processor_local = patch_processor(
        Qwen2VLProcessor.from_pretrained(stg1_local))

    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    wandb.init(project=wandb_project, name=f"avqa_stg2_{run_tag}")

    training_args = SFTConfig(
        output_dir=f"./avqa_stg2_{run_tag}",
        max_steps=stg2["steps"],
        per_device_train_batch_size=stg2["batch_size"],
        gradient_accumulation_steps=stg2["grad_accum"],
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        learning_rate=stg2["lr"],
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=stg2["warmup"],
        lr_scheduler_type=lr_schedule,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
    )

    trainer.train()
    print("  Stage 2 complete")

    def _eval_fn(mdl, samples, split, n=50):
        return evaluate_vqa(
            mdl, processor_local, samples,
            split_name=split, n=n, run_tag=run_tag)

    m_train = _eval_fn(model, eval_train_samples, "train_stg2")
    m_val = _eval_fn(model, eval_val_samples, "val_stg2")
    wandb.log({f"stg2_{k}_train": v for k, v in m_train.items()
               if isinstance(v, (int, float))})
    wandb.log({f"stg2_{k}_val": v for k, v in m_val.items()
               if isinstance(v, (int, float))})

    stg2_local = "./qwen-audio-vqa-ft"
    model.save_pretrained(stg2_local)
    processor_local.save_pretrained(stg2_local)
    model.push_to_hub(hf_repo_ft)
    processor_local.push_to_hub(hf_repo_ft)
    print(f"  Saved to {hf_repo_ft}")
    wandb.finish()

    return m_train, m_val


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL for Audio VQA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_tag = datetime.now().strftime("%m%d%H%M")
    exp_name = cfg.get("exp_name", "")
    if exp_name:
        run_tag = f"{run_tag}_{exp_name}"

    load_dotenv(os.path.expanduser("~/.env"))
    login(token=os.environ.get("HF_TOKEN", os.environ.get("hf", "")))

    api = HfApi()
    api.create_repo(cfg["hf_repo_stg1"], exist_ok=True)
    api.create_repo(cfg["hf_repo_ft"], exist_ok=True)

    ds = load_dataset(cfg["dataset_repo"])
    train_dataset = ds["train"]
    val_dataset = ds["validation"]
    print(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}")

    processor = patch_processor(
        Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"))
    processor.save_pretrained("./qwen2-vl-audio-vqa-processor")

    eval_n = cfg.get("eval_n", 100)
    eval_train_samples = [train_dataset[i]
                          for i in range(min(eval_n, len(train_dataset)))]
    eval_val_samples = [val_dataset[i]
                        for i in range(min(eval_n, len(val_dataset)))]

    collate = make_collate_fn(processor)

    wandb.login(key=os.environ["wandb"])

    stg1_local, m_train_stg1, m_val_stg1 = train_stage1(
        cfg, processor, train_dataset, val_dataset,
        eval_train_samples, eval_val_samples, collate, run_tag)

    m_train_stg2, m_val_stg2 = train_stage2(
        cfg, stg1_local, train_dataset,
        eval_train_samples, eval_val_samples, collate, run_tag)

    print("\nTraining complete")
    for stage, mt, mv in [("Stage 1", m_train_stg1, m_val_stg1),
                          ("Stage 2", m_train_stg2, m_val_stg2)]:
        print(f"  {stage}: train exact={mt['exact_match']:.3f} "
              f"soft={mt['soft_match']:.3f} rouge={mt['rouge_l']:.3f}")
        print(f"  {stage}: val   exact={mv['exact_match']:.3f} "
              f"soft={mv['soft_match']:.3f} rouge={mv['rouge_l']:.3f}")

    results = {
        "run_tag": run_tag, "exp_name": exp_name,
        "dataset": cfg["dataset_repo"], "base_model": cfg["base_model"],
        "stg1": cfg["stg1"], "stg2": cfg["stg2"],
        "stg1_metrics_train": m_train_stg1, "stg1_metrics_val": m_val_stg1,
        "stg2_metrics_train": m_train_stg2, "stg2_metrics_val": m_val_stg2,
    }
    with open(f"results_{run_tag}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
