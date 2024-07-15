import torch
from .module import ShakespeareModule
import torch.nn.functional as F


from .settings import num_chars, block_size, decode


def generate_sentense(model_path, max_tokens=1000):
    model = ShakespeareModule.load_from_checkpoint(
        model_path, num_chars=num_chars, block_size=block_size)  # should be outside loop
    model.eval()
    x = torch.zeros(1, dtype=torch.long)
    for _ in range(max_tokens):
        x_in = x[-1]
        logits = model.model(x_in)
        probs = F.softmax(logits, dim=-1).flatten()
        x_next = torch.multinomial(probs, num_samples=1, replacement=True)
        x = torch.cat([x, x_next])

    sentense = decode(x.tolist())

    print(sentense)


def generate(experiment_name):
    if experiment_name:
        generate_sentense(
            model_path=f"data/logs/GPT/{experiment_name}/checkpoints/best.ckpt", max_tokens=100)
