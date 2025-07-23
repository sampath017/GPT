import tiktoken
import test_settings as s
import pickle
import torch
import numpy as np


class TestDataLoaderLite:
    def __init__(self, split):
        self.B, self.T = s.config["dataset"]["batch_size"], s.config["dataset"]["block_size"]
        assert split in {'train', 'val'}

        shards = sorted(
            [str(p) for p in s.sample10B_data_path.iterdir() if split in p.name])
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if s.ddp_master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def load_tokens(self, filename):
        npt = np.load(filename)
        tokens = torch.tensor(npt, dtype=torch.long)

        return tokens

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * s.ddp_global_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buf[:-1]).reshape(B, T)  # inputs
        y = (buf[1:]).reshape(B, T)  # targets

        # advance the position in the tensor
        self.current_position += B * T * s.ddp_world_size

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * s.ddp_world_size + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * s.ddp_global_rank

        return x.to(s.device), y.to(s.device)
