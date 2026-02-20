import gc
import io
import time

import librosa
import torch
import whisper

WHISPER_N_MELS = 128
SAMPLE_RATE = 16000

AUDIO_PAD_ID = 151658
IMAGE_PAD_ID = 151655
IM_START_ID = 151644

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


def bytes2mel(audio_np):
    audio = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio, n_mels=WHISPER_N_MELS)
    return mel.unsqueeze(0)


def bytes_to_waveform(audio_bytes):
    return librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)[0]


def extract_audio_bytes(msgs):
    for msg in msgs:
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "audio":
                return content.get("audio")
    return None


def extract_image(msgs):
    for msg in msgs:
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "image":
                return content.get("image")
    return None


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


def clear_memory():
    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
