import ast
import io
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset
from google.cloud import texttospeech

VOICES = [
    ("en-US-Neural2-A", texttospeech.SsmlVoiceGender.MALE),
    ("en-US-Neural2-C", texttospeech.SsmlVoiceGender.FEMALE),
    ("en-US-Neural2-D", texttospeech.SsmlVoiceGender.MALE),
    ("en-US-Neural2-F", texttospeech.SsmlVoiceGender.FEMALE),
    ("en-US-Neural2-H", texttospeech.SsmlVoiceGender.FEMALE),
    ("en-US-Neural2-J", texttospeech.SsmlVoiceGender.MALE),
]

OUTPUT_DIR = Path("audio_vqa")
HF_REPO = "hyan/audio-vqa-aokvqa"


def synthesize_one(client, text, voice_idx):
    voice_name, gender = VOICES[voice_idx % len(VOICES)]
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="en-US", name=voice_name, ssml_gender=gender),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000),
    )
    return response.audio_content


def batch_synthesize(questions, cache_dir, max_workers=20):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = texttospeech.TextToSpeechClient()
    results = {}
    to_synthesize = []

    for i, q in enumerate(questions):
        cache_file = cache_dir / f"{i:06d}.wav"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            results[i] = cache_file.read_bytes()
        else:
            to_synthesize.append((i, q))

    print(f"Cached: {len(results)}, to synthesize: {len(to_synthesize)}")
    if not to_synthesize:
        return results

    done = 0
    batch_size = 50
    t0 = time.time()

    for batch_start in range(0, len(to_synthesize), batch_size):
        batch = to_synthesize[batch_start : batch_start + batch_size]
        batch_t0 = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, q in batch:
                futures[executor.submit(synthesize_one, client, q, idx)] = idx

            for f in futures:
                idx = futures[f]
                try:
                    audio_bytes = f.result()
                    results[idx] = audio_bytes
                    (cache_dir / f"{idx:06d}.wav").write_bytes(audio_bytes)
                except Exception as e:
                    print(f"  ERROR on question {idx}: {e}")

        done += len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed * 60 if elapsed > 0 else 0
        print(f"  [{done}/{len(to_synthesize)}] {rate:.0f} req/min, "
              f"elapsed: {elapsed:.0f}s")

        batch_elapsed = time.time() - batch_t0
        min_batch_time = len(batch) / 900 * 60
        if batch_elapsed < min_batch_time:
            time.sleep(min_batch_time - batch_elapsed)

    return results


def pick_best_answer(example):
    choices = example.get("choices", [])
    idx = example.get("correct_choice_idx", None)
    if choices and idx is not None:
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        return choices[idx]

    answers = example.get("direct_answers", [])
    if isinstance(answers, str):
        try:
            answers = ast.literal_eval(answers)
        except (ValueError, SyntaxError):
            answers = []
    if answers:
        return Counter(answers).most_common(1)[0][0]

    return ""


def main():
    ds = load_dataset("HuggingFaceM4/A-OKVQA")
    for split in ds:
        print(f"  {split}: {len(ds[split])} samples")

    output_splits = {}
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        split_ds = ds[split_name]
        print(f"\nTTS for {split_name} ({len(split_ds)} questions)...")

        questions = [ex["question"] for ex in split_ds]
        audio_data = batch_synthesize(questions, OUTPUT_DIR / split_name)

        records = []
        skipped = 0
        for i in range(len(split_ds)):
            if i not in audio_data:
                skipped += 1
                continue

            ex = split_ds[i]
            img_buf = io.BytesIO()
            ex["image"].save(img_buf, format="JPEG")

            records.append({
                "audio_bytes": audio_data[i],
                "audio_sampling_rate": 16000,
                "image": {"bytes": img_buf.getvalue()},
                "question_text": ex["question"],
                "answer": pick_best_answer(ex),
                "question_id": str(ex.get("question_id", i)),
            })

        if skipped:
            print(f"  Skipped {skipped} samples with missing audio")

        features = Features({
            "audio_bytes": Value("binary"),
            "audio_sampling_rate": Value("int32"),
            "image": Image(),
            "question_text": Value("string"),
            "answer": Value("string"),
            "question_id": Value("string"),
        })

        output_splits[split_name] = Dataset.from_list(records, features=features)
        print(f"  {split_name}: {len(output_splits[split_name])} samples")

    dataset_dict = DatasetDict(output_splits)
    dataset_dict.push_to_hub(HF_REPO, private=False)
    print(f"Pushed to hub: {HF_REPO}")

    ds_check = load_dataset(HF_REPO)
    for split in ds_check:
        sample = ds_check[split][0]
        print(f"  {split}[0]: q='{sample['question_text']}', "
              f"a='{sample['answer']}', "
              f"audio_len={len(sample['audio_bytes'])}, "
              f"img={sample['image'].size}")


if __name__ == "__main__":
    main()
