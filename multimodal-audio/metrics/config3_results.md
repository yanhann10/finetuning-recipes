# Config 3 Results

Uses Stage 1 checkpoint from config1.
LoRA: r=8, alpha=16, targets=[q_proj, k_proj, v_proj, o_proj, down_proj], 4-bit nf4

## Stage 2: QLoRA (LR=2e-5, 300 steps)

| Metric               | Value |
| -------------------- | ----- |
| Train Loss (start)   | 0.132 |
| Train Loss (end)     | 0.144 |
| Val Loss (start)     | 0.164 |
| Val Loss (end)       | 0.159 |
| Val WER (50 samples) | 7.9%  |
