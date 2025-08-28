import torch
import torch.nn.functional as F
from finetune import s


class ChatAssistant:
    def __init__(self, model):
        self.model = model
        self.tokenizer = s.enc
        self.device = s.device
        self.history = []  # store conversation turns

    def ask(self, user_msg, max_length=256, top_k=50):
        # add new user message
        self.history.append(f"User: {user_msg}")
        prompt = "\n".join(self.history) + "\nAssistant:"

        # encode tokens
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(
            tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        xgen = tokens
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    logits, _ = self.model(xgen)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # top-k sampling
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

            if xcol.item() == self.tokenizer._special_tokens['<|endoftext|>']:
                break

        decoded = self.tokenizer.decode(xgen[0].tolist())

        # only get the last assistant part
        if "Assistant:" in decoded:
            reply = decoded.split("Assistant:")[-1].strip()
        else:
            reply = decoded

        # store reply in history
        self.history.append(f"Assistant: {reply}")
        return reply
