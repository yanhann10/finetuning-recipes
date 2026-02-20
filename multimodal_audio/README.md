# multimodal_audio

Fine-tuning Qwen2-VL with a whisper audio encoder for multimodal speech understanding.

- **asr/**: Speech-to-text fine-tuning on [speechbrain/LargeScaleASR](https://huggingface.co/datasets/speechbrain/LargeScaleASR) with two-stage training (projector-only then full/QLoRA) and WER evaluation
- **audio_vqa/**: Audio visual question-answering on [A-OKVQA](https://github.com/allenai/aokvqa) with TTS-generated audio questions, and semantic match evaluation with an LLM judge
- **core/**: Audio processing utilities
- **eval/**: Error analysis and taxonomy
