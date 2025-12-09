import numpy as np
import torch

def normalize_batch(batch, target_range=(-1, 1), mean=0):
    """
    Normalize a batch of data (NumPy arrays or PyTorch tensors) to a specified range.

    Parameters:
        batch (list or array): A batch of data (NumPy array or PyTorch tensor).
        target_range (tuple): The desired range for normalization (default: (-1, 1)).
        mean (float): Optional mean value to center the data around (default: 0).

    Returns:
        Normalized batch in the same format as the input (NumPy or PyTorch).
    """
    min_val, max_val = target_range

    if isinstance(batch, torch.Tensor):
        batch_min = batch.min(dim=-1, keepdim=True).values
        batch_max = batch.max(dim=-1, keepdim=True).values
        normalized = (batch - batch_min) / (batch_max - batch_min)  # Scale to [0, 1]
        normalized = normalized * (max_val - min_val) + min_val  # Scale to target range
        if mean != 0:
            normalized = normalized - normalized.mean(dim=-1, keepdim=True) + mean
        return normalized

    elif isinstance(batch, np.ndarray):
        batch_min = np.min(batch, axis=-1, keepdims=True)
        batch_max = np.max(batch, axis=-1, keepdims=True)
        normalized = (batch - batch_min) / (batch_max - batch_min)  # Scale to [0, 1]
        normalized = normalized * (max_val - min_val) + min_val  # Scale to target range
        if mean != 0:
            normalized = normalized - np.mean(normalized, axis=-1, keepdims=True) + mean
        return normalized

    else:
        raise TypeError("Input batch must be a NumPy array or PyTorch tensor.")