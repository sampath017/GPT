import torch
from module import NamesModule
import torch.nn.functional as F


from settings import num_chars, block_size, ctoi, itoc, start_char, end_char


def generate_names(model_path, num_names=5):
    for _ in range(num_names):
        model = NamesModule.load_from_checkpoint(
            model_path, num_chars=num_chars, block_size=block_size)
        model.eval()

        out = []
        context = [ctoi[start_char]] * block_size
        while True:
            embs = model.embedding_table(torch.tensor(context).reshape(1, -1))
            embs = embs.reshape(embs.shape[0], -1)
            logits = model.model(embs)
            probs = F.softmax(logits, dim=-1).flatten()
            ix = torch.multinomial(probs, num_samples=1,
                                   replacement=True).item()
            if ix == ctoi[end_char]:
                break

            out.append(ix)

        name = ''.join(itoc[i] for i in out)
        print(name)


def generate():
    generate_names(
        model_path="data\\logs\\Makemore\\u1365m8v\\checkpoints\\epoch=7019-step=63180.ckpt", num_names=1)
