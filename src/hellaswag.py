"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import json
import requests
from tqdm import tqdm
import torch
from torch.nn import functional as F
import settings as s
import torch.distributed as dist
from transformers import GPT2LMHeadModel


def download_file(url, fname, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with fname.open("wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    data_url = hellaswags[split]
    data_filename = s.data_root_path / f"hellaswag/hellaswag_{split}.jsonl"
    if not data_filename.exists():
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = s.enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        # note: prepending " " because GPT-2 tokenizer
        end_tokens = s.enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(s.data_root_path / f"hellaswag/hellaswag_{split}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def get_most_likely_row(logits, y, mask):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_y = y.reshape(-1)
    shift_losses = F.cross_entropy(
        flat_logits, flat_y, reduction='none')

    shift_losses = shift_losses.reshape(y.shape[0], -1)

    # now get the average loss just for the completion region (where mask == 1), in each row
    # we must shift mask, so we start at the last prompt token
    shift_mask = mask[..., 1:]
    masked_shift_losses = shift_losses * shift_mask

    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()

    return pred_norm


@torch.no_grad()
def evaluate(model):
    # once in a while evaluate hellaswag
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % s.ddp_world_size != s.ddp_local_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(s.device)
        mask = mask.to(s.device)
        x = tokens[..., :-1]
        y = tokens[..., 1:]

        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
                if isinstance(model, GPT2LMHeadModel):
                    logits = model(x).logits
                else:
                    logits, loss = model(x)
            pred_norm = get_most_likely_row(logits, y, mask)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce the stats across all processes
    if s.is_ddp_available:
        num_total = torch.tensor(num_total, dtype=torch.long, device=s.device)
        num_correct_norm = torch.tensor(
            num_correct_norm, dtype=torch.long, device=s.device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()

    hellaswag_acc = num_correct_norm / num_total
    if s.ddp_master_process:
        print(
            f"HellaSwag accuracy: {num_correct_norm}/{num_total}={hellaswag_acc:.4f}")

    return hellaswag_acc
