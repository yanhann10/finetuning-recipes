# Config 2 Results

## Stage 1: Projector Only (22,500 steps)

| Metric                 | Value |
| ---------------------- | ----- |
| Train Loss (start)     | 3.235 |
| Train Loss (end)       | 0.095 |
| Val Loss               | N/A   |
| Token Accuracy (start) | 42.5% |
| Token Accuracy (end)   | 97.9% |
| Train WER (50 samples) | 7.1%  |
| Val WER (50 samples)   | 8.9%  |

## Stage 2: QLoRA (500 steps, best at step 350)

| Metric                     | Value  |
| -------------------------- | ------ |
| Val Loss (start)           | 0.1663 |
| Val Loss (end)             | 0.1649 |
| Best Val WER (50 samples)  | 7.6%   |
| Final Val WER (50 samples) | 7.8%   |
