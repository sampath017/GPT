# GPT

## Training
uv run torchrun --standalone --nproc-per-node=2 src/main.py

## Training steps

step 5   | train_loss 7.83  | val_loss 7.81  | norm 0.75  | time 1297.64 ms
step 6   | train_loss 7.91  | val_loss 7.79  | norm 0.57  | time 1303.11 ms
step 7   | train_loss 7.80  | val_loss 7.86  | norm 0.86  | time 1300.03 ms
step 8   | train_loss 7.87  | val_loss 7.89  | norm 0.91  | time 1304.53 ms
