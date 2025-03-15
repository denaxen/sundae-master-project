"""Utilities.

Some functions copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
"""

import json
import os

import fsspec
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf


def compute_loss(logits, targets):
    # logits: tensor of shape (batch_size, seq_len, vocab_size)
    # targets: tensor of shape (batch_size, seq_len)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum"
    )
    return loss


def add_resolvers():
    OmegaConf.register_new_resolver("cwd", os.getcwd)
    OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def prepare_logger():
    fn_names = ["trace", "debug", "info", "success", "warning", "error", "critical"]
    for k in fn_names:
        fn = getattr(logger, k)
        fn = L.pytorch.utilities.rank_zero_only(fn)
        setattr(logger, k, fn)


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def params2key(**kwargs):
    bool_entries = []
    non_bool_entries = []

    for k, v in kwargs.items():
        # Replace / in path so that it doesn't create sub-directories
        if type(v) is str:
            v = v.replace("/", "#")
        if type(v) is bool:
            bool_entries.append((k, v))
        else:
            non_bool_entries.append((k, v))

    bool_entries.sort(key=lambda x: x[0])
    non_bool_entries.sort(key=lambda x: x[0])

    name = ""
    for k, v in non_bool_entries:
        name += str(k) + "=" + str(v)
        name += ","

    for i, (k, v) in enumerate(bool_entries):
        if v:
            name += k
        else:
            name += "no_" + k

        if i != len(bool_entries) - 1:
            name += ","

    return name


def find_value_token_indices(tokens, value_string, enc):
    decoded_tokens = [enc.decode([token]) for token in tokens]
    reconstructed_string = "".join(decoded_tokens)

    # Find start and end indices of value_string in reconstructed_string
    start_index = reconstructed_string.find(value_string)
    if start_index == -1:
        # Value string not found in reconstructed string
        return []

    end_index = start_index + len(value_string)

    # Build token positions
    token_positions = []
    pos = 0
    for i, token in enumerate(decoded_tokens):
        token_length = len(token)
        token_start = pos
        token_end = pos + token_length  # Exclusive end index
        token_positions.append((token_start, token_end, i))
        pos = token_end

    # Find tokens that overlap with value_string positions
    value_token_indices = []
    for token_start, token_end, idx in token_positions:
        if token_end <= start_index:
            continue  # Token ends before value_string starts
        elif token_start >= end_index:
            continue  # Token starts after value_string ends
        else:
            # Token overlaps with value_string
            value_token_indices.append(idx)
    start, end = min(value_token_indices), max(value_token_indices)
    return start, end


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)