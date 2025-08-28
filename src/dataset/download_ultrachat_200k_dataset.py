import sys
from datasets import load_dataset
import os
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
sys.path.append(Path(__file__).parent.parent.as_posix())  # nopep8
import finetune.settings as s

data_root_path = s.data_root_path/"ultrachat_200k"
data_root_path.mkdir(exist_ok=True, parents=True)

# setup
eot = s.enc._special_tokens['<|endoftext|>']  # end of text token
shard_size = int(50e6)  # 50 million tokens per shard, total of 6 shards


def convo_to_str(example):
    # flatten one conversation
    return "\n".join([f"{m['role']}: {m['content']}" for m in example["messages"]])


# SFT supervised training split
ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                  cache_dir=s.data_root_path.as_posix(), split="train_sft")
ds = ds.map(lambda x: {"text": convo_to_str(x)})

# setup
data_root_path = s.data_root_path/"ultrachat_200k"
data_root_path.mkdir(exist_ok=True)
eot = s.enc._special_tokens['<|endoftext|>']  # end of text token
shard_size = int(5e7)  # 50 million tokens per shard, total of 6 shards


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(s.enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 **
                                       16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)

    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count())  # type: ignore
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, ds, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = data_root_path / \
                f"{split}_{shard_index:02d}"
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)  # type: ignore
            all_tokens_np[token_count:token_count +
                          remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    # TODO measure the count of val and train tokens per shard.
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = data_root_path / f"{split}_{shard_index:02d}"
        write_datafile(filename, all_tokens_np[:token_count])
