# Config 1 Results

## Stage 1: Projector Only (2200 steps, LR=3e-4, GA=2)

| Metric           | Value |
| ---------------- | ----- |
| Val Loss (start) | 0.943 |
| Val Loss (end)   | 0.179 |
| Token Accuracy   | 96%   |
| Train WER        | 8.67% |
| Val WER          | 5.88% |

## Stage 2: Full Fine-tuning (300 steps, LR=2e-5, GA=4, early stopped)

| Metric           | Value  |
| ---------------- | ------ |
| Val Loss (start) | 0.178  |
| Val Loss (end)   | 0.185  |
| Train Loss       | 0.153  |
| Train WER        | 11.67% |
| Val WER          | 6.62%  |
