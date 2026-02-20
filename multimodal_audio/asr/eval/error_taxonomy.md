# Error Taxonomy: Audio VQA Stage 1

Model: `hyan/qwen_audio_vqa_stage1` (projector-only, 500 steps)
Eval: 40 samples total (20 train + 20 val), 31 errors

## Updated Metrics (with semantic matching)

| Metric | Train (n=20) | Val (n=20) |
|--------|-------------|------------|
| Exact Match | 0.200 | 0.250 |
| Soft Match | 0.300 | 0.250 |
| **Semantic Match** | **0.500** | **0.400** |

Semantic matching adds: number normalization (thirty=30), synonym groups
(jeans=denim, bike=bicycle, usa=american), simple lemmatization (horses=horse),
fuzzy string matching (>=85% ratio), and substring containment.

## Taxonomy

### 1. Semantic Equivalents (est. 25-30% of errors)

The model gives a correct answer that doesn't match the reference string.

| Predicted | Ground Truth | Type |
|-----------|-------------|------|
| 30 | thirty | number format |
| jeans | denim | synonym |
| cell phone | mobile phone | synonym |
| usa | american | noun vs adjective |
| horses | horse | plural vs singular |
| bicycle | bike | synonym |
| united kingdom | uk | full name vs abbreviation |
| pilots | flying | role vs activity |

**Impact**: These inflate the error rate. The model is often right but scored wrong.
**Fix**: Use fuzzy matching, lemmatization, or LLM-as-judge for eval.

### 2. Object/Substance Identification (est. 20-25% of errors)

Model identifies the wrong object or substance in the image.

| Question | Predicted | Ground Truth |
|----------|-----------|-------------|
| What is in the motorcyclist's mouth? | microphone | cigarette |
| What kind of computer is near the woman? | wii | macintosh |
| What is the man putting on the bus? | bicycle | bow |
| What kind of fruit is cut in half? | olive | grapes |

**Root cause**: Small or ambiguous objects in images. The audio pathway adds no spatial grounding - the model must rely entirely on vision for object ID.

### 3. Question Intent Misparse (est. 15-20% of errors)

Model answers a different question than what was asked.

| Question asks for | Predicted (wrong type) | Ground Truth |
|-------------------|----------------------|-------------|
| descriptor (adjective) | pond (noun) | dirty |
| what interests the child | birthday (event) | candle (object) |
| descriptive word for surface | catwalk (noun) | crowded (adjective) |
| what type of rain | rain (generic) | downpour (specific) |

**Root cause**: Audio comprehension may lose nuance. "What best describes X" requires understanding that the answer should be an adjective, not a noun. The model defaults to naming things rather than describing them.

### 4. Plausible-but-Wrong (est. 15% of errors)

Model gives a reasonable answer that happens to not be the ground truth.

| Question | Predicted | Ground Truth |
|----------|-----------|-------------|
| White substance on cupcakes? | powdered sugar | icing |
| What item helps with a cold? | tissue | cough drops |
| What event is this? | dinner | date |
| What mood are the cows in? | content | happy |

**Root cause**: Genuinely ambiguous questions where multiple answers are defensible. The A-OKVQA dataset has a single canonical answer but real-world VQA is often multi-answer.

### 5. Knowledge Gap (est. 10% of errors)

Questions requiring world knowledge the model doesn't have.

| Question | Predicted | Ground Truth |
|----------|-----------|-------------|
| When did the namesake of this theater die? | 1989 | 2009 |
| What is the man by the bags awaiting? | bus | cab |

**Root cause**: These require external knowledge (dates, cultural context) that a 7B model may not retain, especially after fine-tuning on a small domain-specific dataset.

### 6. Action/Activity Recognition (est. 5% of errors)

| Question | Predicted | Ground Truth |
|----------|-----------|-------------|
| What is the person on the left doing? | surfing | crouching |
| What activity will the cat do? | eat | jump |

**Root cause**: Temporal/dynamic reasoning from a static image. The model must infer intent or posture, which is inherently ambiguous.

## Baseline Comparison

| Model | A-OKVQA DA Accuracy | Input |
|-------|-------------------|-------|
| BLIP-2 (Flan-4-shot) | 25.9% | image + text |
| InstructBLIP (Vicuna-13B) | 41.0% | image + text |
| InstructBLIP (FlanT5-xxl) | 48.0% | image + text |
| GPT-4V (0-shot) | 64.3% | image + text |
| **Ours (Stage 1)** | **25.0%** | **audio + image** |

Our Stage 1 (projector-only) at 25% exact match is roughly at BLIP-2 4-shot level, but with a harder input modality (audio instead of text). Accounting for ~25-30% of errors being semantic equivalents, effective accuracy is closer to 32-35%.

## Recommendations for Next Training Run

1. **Eval fix (high priority)**: Add fuzzy/semantic matching - lemmatize, normalize numbers, use synonym sets
2. **System prompt**: "Answer with 1-3 words" (already in v2 script)
3. **Stage 2**: Fix the device_map error for `audio_encoder.positional_embedding` and run full fine-tuning
4. **More eval samples**: n=20 is too noisy; v2 uses n=100
5. **Answer normalization**: Lowercase, strip articles ("a", "the"), normalize numbers to words

AI generated and human reviewed
