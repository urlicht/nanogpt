import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

def _sample_batch(data: np.memmap, n_batch: int, n_block: int):
    max_start = len(data) - n_block - 1
    if max_start <= 0:
        raise ValueError("dataset too small for the requested block size")
    ix = torch.randint(max_start, (n_batch,)).numpy()
    offsets = np.arange(n_block)
    idx = ix[:, None] + offsets[None, :]
    x = torch.from_numpy(data[idx].astype(np.int64))
    y = torch.from_numpy(data[idx + 1].astype(np.int64))
    
    return x, y

def get_batch(
    data_dir: str|Path, 
    split: str, 
    n_batch: int, 
    n_block: int, 
    device: torch.device=torch.device('cpu'), 
    dtype: type=np.uint16
):
    # re-open the memmap every batch to avoid the memory 'leak' (cache buildup)
    filename = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(filename, dtype=dtype, mode='r')
    
    x, y = _sample_batch(data, n_batch, n_block)
    del data
    
    # CUDA optimization
    if device.type == 'cuda':
        # pin_memory + non_blocking to speed up VRAM transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

class MemmapBatchDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str|Path,
        split: str,
        n_batch: int,
        n_block: int,
        dtype: type=np.uint16,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.split = split
        self.n_batch = n_batch
        self.n_block = n_block
        self.dtype = dtype

    def __iter__(self):
        filename = os.path.join(self.data_dir, f"{self.split}.bin")
        while True:
            # re-open the memmap every batch to avoid the memory 'leak' (cache buildup)
            data = np.memmap(filename, dtype=self.dtype, mode="r")
            x, y = _sample_batch(data, self.n_batch, self.n_block)
            del data
            yield x, y

def build_dataloader(
    data_dir: str|Path,
    split: str,
    n_batch: int,
    n_block: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    dataset = MemmapBatchDataset(
        data_dir=data_dir,
        split=split,
        n_batch=n_batch,
        n_block=n_block,
    )
    loader_kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
