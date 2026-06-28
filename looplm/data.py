"""FineWeb-style data loader. Reads uint16 .bin shards of GPT-2 tokens.

Format matches Karpathy's llm.c: each .bin is a flat uint16 array. Shards are
sharded across DDP ranks; each rank streams its own non-overlapping slice.

Adapted from rootxhacker/HobbyLM (Apache-2.0).
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _load_shard(path: str) -> np.ndarray:
    """Memory-map a uint16 .bin shard. The first 1024 bytes are an llm.c header
    in some shards; we skip it if the magic matches, otherwise read from offset 0."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(1024), dtype=np.int32)
    # llm.c header magic = 20240520
    offset = 256 if len(header) >= 1 and header[0] == 20240520 else 0
    arr = np.memmap(path, dtype=np.uint16, mode="r", offset=offset * 4 if offset else 0)
    return arr


def resolve_shards(pattern_or_shards: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(pattern_or_shards, (str, Path)):
        shards = sorted(glob.glob(str(pattern_or_shards)))
    else:
        shards = [str(path) for path in pattern_or_shards]
    if not shards:
        raise FileNotFoundError(f"no shards matched: {pattern_or_shards}")
    return shards


def data_generator(pattern_or_shards: str | Path | Sequence[str | Path], B: int, S: int, device,
                   rank: int = 0, world: int = 1, to_device: bool = True):
    """Infinite generator of (x, y) batches.

    pattern_or_shards: glob like "/data/fineweb_edu/edu_fineweb_train_*.bin"
      or an explicit ordered shard list.
    B: micro batch size (sequences)
    S: sequence length (tokens)
    rank/world: DDP sharding; each rank gets non-overlapping slices.
    """
    shards = resolve_shards(pattern_or_shards)
    tokens_per_batch = B * S
    shard_idx = 0
    pos = rank * tokens_per_batch  # stagger ranks
    while True:
        arr = _load_shard(shards[shard_idx])
        # we need (B*S + 1) tokens to form (x, y) with y shifted by 1
        while pos + tokens_per_batch + 1 <= len(arr):
            buf = torch.from_numpy(arr[pos:pos + tokens_per_batch + 1].astype(np.int64))
            x = buf[:-1].view(B, S)
            y = buf[1:].view(B, S)
            if to_device:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            yield x, y
            pos += tokens_per_batch * world  # advance by world batches so ranks don't overlap
        shard_idx = (shard_idx + 1) % len(shards)
        pos = rank * tokens_per_batch


class CUDAPrefetcher:
    """Overlap H2D copy with compute. Holds one batch ahead on the GPU stream."""

    def __init__(self, gen, device):
        self.gen = gen
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._preload()

    def _preload(self):
        try:
            x, y = next(self.gen)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
        self.next_batch = (x, y)

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch
