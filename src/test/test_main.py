import os
import torch
import torch.distributed as dist
from test_dataset import TestDatasetLite, TestDataLoaderLite
import test_settings as s

dataset = TestDatasetLite()
dataloader = TestDataLoaderLite(dataset, split="train")

for rank in range(s.ddp_world_size):
    dist.barrier()  # sync all processes
    if dist.get_rank() == rank:
        print(
            f"rank={rank} of position: {dataloader.current_position}", flush=True)
