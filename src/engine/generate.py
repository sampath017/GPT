import torch
from .module import ShakespeareModule
import torch.nn.functional as F

import shutil
from pathlib import Path

from .settings import num_chars, context_size, decode, models_path


def generate_sentense(model_path, max_tokens=1000):
    model = ShakespeareModule.load_from_checkpoint(
        model_path, num_chars=num_chars, context_size=context_size)  # should be outside loop
    model.eval()
    x = torch.zeros(context_size, dtype=torch.long)
    for _ in range(max_tokens):
        x_in = x[-context_size:]
        logits = model.model(x_in)
        last_token = logits[-1]
        probs = F.softmax(last_token, dim=-1).flatten()
        x_next = torch.multinomial(probs, num_samples=1, replacement=True)
        x = torch.cat([x, x_next])

    sentense = decode(x[context_size:].tolist())

    print(sentense)

    copy_file_to_models(model_path, models_path / "best.ckpt")


def copy_file_to_models(source, destination):
    shutil.copy(source, destination)

    print(f"File copied from {source} to {destination}")


def generate(experiment_name):
    if experiment_name:
        generate_sentense(
            model_path=f"data/logs/GPT/{experiment_name}/checkpoints/best.ckpt", max_tokens=100)
