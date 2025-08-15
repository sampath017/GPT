# GPT

## Training
uv run torchrun --standalone --nproc-per-node=2 src/main.py

## Training steps

step 5   | train_loss 7.83  | val_loss 7.81  | norm 0.75  | time 1297.64 ms
step 6   | train_loss 7.91  | val_loss 7.79  | norm 0.57  | time 1303.11 ms
step 7   | train_loss 7.80  | val_loss 7.86  | norm 0.86  | time 1300.03 ms
step 8   | train_loss 7.87  | val_loss 7.89  | norm 0.91  | time 1304.53 ms

# TO-UNDERSTAND
1. 
    why reverse weight sharing scheme is resulting in large loss ?
    step 0   | train_loss 472.79 | norm 95.00 | time 54479.77 ms | tok/sec 9.62K
    step 1   | train_loss 121.85 | norm 286.41 | time 48853.12 ms | tok/sec 10.73K
    step 2   | train_loss 106.30 | norm 127.80 | time 48945.54 ms | tok/sec 10.71K


# Previous A100 - 2 RUN
step 196 | train_loss 7.74  | norm 0.24  | time 1314.65 ms | tok/sec 398.81K
step 197 | train_loss 7.71  | norm 0.18  | time 1316.37 ms | tok/sec 398.28K
step 198 | train_loss 7.66  | norm 0.25  | time 1315.80 ms | tok/sec 398.46K
step 199 | train_loss 7.67  | norm 0.38  | time 1313.40 ms | tok/sec 399.18K
