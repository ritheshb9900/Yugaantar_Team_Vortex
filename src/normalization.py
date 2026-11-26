"""Channel normalization and denormalization using official competition statistics."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Union
import numpy as np
import torch

STATS_FILE = Path(__file__).resolve().parent.parent / "stats.json"


def load_stats(stats_path: Union[str, Path] = STATS_FILE) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load channel means, stds, and channel names from stats.json."""
    stats_path = Path(stats_path).resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    channels = data["channels"]
    mean = np.array(data["mean"], dtype=np.float32)
    std = np.array(data["std"], dtype=np.float32)
    return mean, std, channels


class ChannelNormalizer:
    """Normalizes and denormalizes the 6 dynamic atmospheric variables.
    
    Formula:
        Z = (X - mean) / std
        X = Z * std + mean
    """

    def __init__(self, stats_path: Union[str, Path] = STATS_FILE):
        self.mean_np, self.std_np, self.channels = load_stats(stats_path)
        self.mean_torch = torch.from_numpy(self.mean_np)
        self.std_torch = torch.from_numpy(self.std_np)

    def normalize(self, x: Union[np.ndarray, torch.Tensor], channel_axis: int = -3) -> Union[np.ndarray, torch.Tensor]:
        """Normalize physical weather array/tensor into dimensionless z-scores."""
        if isinstance(x, torch.Tensor):
            mean = self.mean_torch.to(device=x.device, dtype=x.dtype)
            std = self.std_torch.to(device=x.device, dtype=x.dtype)
            # Reshape mean and std to broadcast with x
            shape = [1] * x.ndim
            shape[channel_axis] = -1
            return (x - mean.view(*shape)) / std.view(*shape)
        elif isinstance(x, np.ndarray):
            shape = [1] * x.ndim
            shape[channel_axis] = -1
            return (x - self.mean_np.reshape(*shape)) / self.std_np.reshape(*shape)
        else:
            raise TypeError(f"Unsupported data type: {type(x)}")

    def denormalize(
        self,
        z: Union[np.ndarray, torch.Tensor],
        channel_axis: int = -3,
        clamp_physical: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize dimensionless z-scores back to raw physical units."""
        if isinstance(z, torch.Tensor):
            mean = self.mean_torch.to(device=z.device, dtype=z.dtype)
            std = self.std_torch.to(device=z.device, dtype=z.dtype)
            shape = [1] * z.ndim
            shape[channel_axis] = -1
            x = z * std.view(*shape) + mean.view(*shape)
            if clamp_physical:
                # Clamp precipitation >= 0
                tp_slice = [slice(None)] * z.ndim
                tp_slice[channel_axis] = 1
                x[tuple(tp_slice)] = torch.clamp(x[tuple(tp_slice)], min=0.0)
                # Clamp temperature [-40C, +60C] -> [233.15K, 333.15K]
                t2m_slice = [slice(None)] * z.ndim
                t2m_slice[channel_axis] = 0
                x[tuple(t2m_slice)] = torch.clamp(x[tuple(t2m_slice)], min=233.15, max=333.15)
            return x
        elif isinstance(z, np.ndarray):
            shape = [1] * z.ndim
            shape[channel_axis] = -1
            x = z * self.std_np.reshape(*shape) + self.mean_np.reshape(*shape)
            if clamp_physical:
                tp_slice = [slice(None)] * z.ndim
                tp_slice[channel_axis] = 1
                x[tuple(tp_slice)] = np.maximum(x[tuple(tp_slice)], 0.0)
                t2m_slice = [slice(None)] * z.ndim
                t2m_slice[channel_axis] = 0
                x[tuple(t2m_slice)] = np.clip(x[tuple(t2m_slice)], 233.15, 333.15)
            return x
        else:
            raise TypeError(f"Unsupported data type: {type(z)}")
