import glob

import os
import sys
import traceback

import librosa
import torch
import whisper
from transformers import Qwen2VLForConditionalGenerationWithAudio, Qwen2VLProcessor

WHISPER_N_MELS = 128
SAMPLE_RATE = 16000
CHUNK_SEC = 25

MODELS = [
    "hyan/qwen-speech-ft_cfg1",
    "hyan/qwen-speech-ft",
]

orig = (
    '{% else %}{% for content in message["content"] %}'
    '{% if content["type"] == "image" or "image" in content or "image_url" in content %}'
    '{% set image_count.value = image_count.value + 1 %}'
    '{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}'
    '<|vision_start|><|image_pad|><|vision_end|>'
    '{% elif content["type"] == "video" or "video" in content %}'
    '{% set video_count.value = video_count.value + 1 %}'
    '{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}'
    '<|vision_start|><|video_pad|><|vision_end|>'
    '{% elif "text" in content %}{{ content["text"] }}'
    '{% endif %}{% endfor %}<|im_end|>'
)

replacements = (
    '{% else %}{% for content in message["content"] %}'
    '{% if content["type"] == "audio" or "audio" in content or "audio_url" in content %}'
    '<|audio_start|><|audio_pad|><|audio_end|>'
    '{% elif content["type"] == "image" or "image" in content or "image_url" in content %}'
    '{% set image_count.value = image_count.value + 1 %}'
    '{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}'
    '<|vision_start|><|image_pad|><|vision_end|>'
    '{% elif content["type"] == "video" or "video" in content %}'
    '{% set video_count.value = video_count.value + 1 %}'
    '{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}'
    '<|vision_start|><|video_pad|><|vision_end|>'
    '{% elif "text" in content %}{{ content["text"] }}'
    '{% endif %}{% endfor %}<|im_end|>'
)


def patch_processor(proc):
    special_tokens = ["<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"]
    proc.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tmpl = proc.tokenizer.chat_template
    if "audio" not in tmpl:
        tmpl = tmpl.replace(orig, replacements, 1)
        proc.tokenizer.chat_template = tmpl
    proc.chat_template = proc.tokenizer.chat_template
    return proc


def np_to_mel(audio_np):
    audio = whisper.pad_or_trim(audio_np)
    return whisper.log_mel_spectrogram(audio, n_mels=WHISPER_N_MELS)


def chunk_waveform(audio_np, chunk_sec):
    if chunk_sec <= 0 or len(audio_np) <= chunk_sec * SAMPLE_RATE:
        return [audio_np]
    chunk_len = chunk_sec * SAMPLE_RATE
    return [audio_np[start:start + chunk_len]
            for start in range(0, len(audio_np), chunk_len)]


def transcribe_single(model, processor, mel):
    user_msgs = [{"role": "user", "content": [
        {"type": "audio", "audio": b"placeholder"},
        {"type": "text", "text": "Transcribe this audio."},
    ]}]
    text = processor.apply_chat_template(
        user_msgs, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], audio=[mel], padding=True, return_tensors="pt")
    batch = {k: v.to(model.device) if hasattr(v, "to") else v
             for k, v in batch.items()}
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=256, do_sample=False)
    return processor.tokenizer.decode(
        gen[0][batch["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()


def transcribe(model, processor, wav_path, chunk_sec=CHUNK_SEC):
    audio_np, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    chunks = chunk_waveform(audio_np, chunk_sec)
    parts = []
    for chunk in chunks:
        mel = np_to_mel(chunk).cpu()
        parts.append(transcribe_single(model, processor, mel))
    return " ".join(parts)


def load_model(hf_repo):
    model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(
        hf_repo, torch_dtype=torch.bfloat16, device_map="auto")
    processor = patch_processor(Qwen2VLProcessor.from_pretrained(hf_repo))
    model.audio_encoder.float()
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    model.eval()
    return model, processor


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_files = sorted(glob.glob(os.path.join(script_dir, "*.wav")))
    if not wav_files:
        print("No .wav files found in testing/")
        sys.exit(1)

    results = []
    for hf_repo in MODELS:
        try:
            model, processor = load_model(hf_repo)
        except Exception:
            traceback.print_exc()
            for wav_path in wav_files:
                results.append((os.path.basename(wav_path), "MODEL_LOAD_ERROR", hf_repo))
            continue

        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            try:
                hyp = transcribe(model, processor, wav_path)
                print(f"[{hf_repo}] {fname}: {hyp[:80]}")
            except Exception:
                traceback.print_exc()
                hyp = "TRANSCRIPTION_ERROR"
            results.append((fname, hyp, hf_repo))

        del model, processor
        torch.cuda.empty_cache()

    out_path = os.path.join(script_dir, "generated_transcript.txt")
    with open(out_path, "w") as f:
        f.write("filename\ttranscript\thf_repo\n")
        for fname, hyp, repo in results:
            hyp_clean = hyp.replace("\t", " ").replace("\n", " ")
            f.write(f"{fname}\t{hyp_clean}\t{repo}\n")
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
