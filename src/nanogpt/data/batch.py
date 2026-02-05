import os
import torch
import numpy as np

def get_batch(
    data_dir: str, 
    split: str, 
    n_batch: int, 
    n_block: int, 
    device: torch.device=torch.device('cpu'), 
    dtype: type=np.uint16
):
    # re-open the memmap every batch to avoid the memory 'leak' (cache buildup)
    filename = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(filename, dtype=dtype, mode='r')
    
    # randomly sample starting indices for the batch
    # We use len(data) - n_block - 1 to ensure we have room for the y-offset
    ix = torch.randint(len(data) - n_block - 1, (n_batch,))
    
    # extract slices and stack them
    # torch requires int64
    x = torch.stack([torch.from_numpy((data[i : i + n_block]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + n_block]).astype(np.int64)) for i in ix])
    
    # CUDA optimization
    if device.type == 'cuda':
        # pin_memory + non_blocking to speed up VRAM transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y