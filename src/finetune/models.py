import torch
import torch.nn as nn
import torch.nn.functional as F
from finetune import s
from finetune.utils import ModelSummary

vocab_size = s.config["dataset"]["vocab_size"]
block_size = s.config["dataset"]["block_size"]
num_embds = s.config["model"]["num_embds"]
head_size = s.config["model"]["head_size"]
num_heads = s.config["model"]["num_heads"]
num_blocks = s.config["model"]["num_blocks"]
dropout = s.config["model"]["dropout"]
batch_size = s.config["dataset"]["batch_size"]


# Masked attention
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert num_embds % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(num_embds, 3 * num_embds)
        # output projection
        self.proj = nn.Linear(num_embds, num_embds)
        self.proj.NANOGPT_SCALE_INIT = 1  # type: ignore
        # regularization
        self.n_head = num_heads
        self.n_embd = num_embds

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True)  # flash attention
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(num_embds, num_embds*4),
            nn.GELU(),
            nn.Linear(num_embds*4, num_embds),
            nn.Dropout(dropout),
        )

    def forward(self, affinities):
        return self.ff(affinities)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(num_embds)
        self.self_attention = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(num_embds)
        self.feed_forward = FeedForward()

    def forward(self, embds):
        embds = embds + self.self_attention(self.ln1(embds))
        affinities = embds + self.feed_forward(self.ln2(embds))

        return affinities


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            token_embedding_table=nn.Embedding(
                vocab_size, num_embds),  # type: ignore
            position_embedding_table=nn.Embedding(block_size, num_embds),
            blocks=nn.Sequential(*[Block() for _ in range(num_blocks)]),
            ln_f=nn.LayerNorm(num_embds)
        ))
        self.lm_head = nn.Linear(num_embds, vocab_size)

        # weight sharing scheme
        self.lm_head.weight = self.transformer.token_embedding_table.weight  # type: ignore

        self.apply(self._init_weights)

    def forward(self, x, y=None):
        B, T = x.shape

        token_embds = self.transformer.token_embedding_table(
            x)  # type: ignore # (B, T, num_embds)
        pos_embds = self.transformer.position_embedding_table(
            # type: ignore # (T, num_embds)
            torch.arange(T, device=s.config["device"]))
        embds = token_embds + pos_embds  # (B, T, num_embds)

        affinities = self.transformer.ln_f(
            self.transformer.blocks(embds))  # type: ignore
        logits = self.lm_head(affinities)  # (B, T, vocab_size)

        loss = None
        if y is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas, eps):
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]  # nopep8
        non_decay_params = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]  # nopep8

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]

        if s.ddp_master_process:
            num_decay_params = sum(p.numel() for p in decay_params)
            num_non_decay_params = sum(p.numel() for p in non_decay_params)
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {ModelSummary.format_number(num_decay_params)} parameters")
            print(
                f"num non-decayed parameter tensors: {len(non_decay_params)}, with {ModelSummary.format_number(num_non_decay_params)} parameters")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, eps=eps, fused=True)

        return optimizer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * num_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
