"""PyTorch Dataset and DataLoader for 3-day spatiotemporal sequence forecasting."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from monsoon_r1.contract import doy_sincos, CHANNELS, NC, NLAT, NLON
from src.normalization import ChannelNormalizer

# Chronological split definitions (inclusive date ranges)
SPLIT_RANGES = {
    "train": (20050101, 20181231),
    "val":   (20190101, 20201231),
    "test":  (20210101, 20211231),
}


def _date_to_doy(date_int: int) -> int:
    """Convert integer YYYYMMDD to calendar day of year (1-366)."""
    s = str(int(date_int))
    d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
    return d.timetuple().tm_yday


class ClimateSequenceDataset(Dataset):
    """Spatiotemporal sequence dataset for numerical monsoon forecasting.

    Input shape per sample:  (22, 32, 36)
      - 18 channels: 3 consecutive days x 6 normalized dynamic variables
      - 2 channels:  static features [land_sea_mask, orography / 1000.0]
      - 2 channels:  spatially broadcast DOY [sin, cos] for the LAST input day (Day t)

    Target shape per sample: (6, 32, 36)
      - 6 channels: normalized dynamic variables for Day t+1
    """

    def __init__(
        self,
        split: Literal["train", "val", "test", "all"] = "train",
        cache_path: str | Path = "data/cache/climate_data.npz",
        artifacts_dir: str | Path = "artifacts",
        normalizer: Optional[ChannelNormalizer] = None,
        t_in: int = 3,
        step_ahead: int = 1,
    ):
        super().__init__()
        self.split = split
        self.t_in = t_in
        self.step_ahead = step_ahead
        self.normalizer = normalizer or ChannelNormalizer()

        cache_path = Path(cache_path).resolve()
        artifacts_dir = Path(artifacts_dir).resolve()

        if not cache_path.is_file():
            raise FileNotFoundError(f"Cached data not found: {cache_path}. Run scripts/extract_data.py first.")

        # Load cached dynamic data & dates
        z = np.load(cache_path)
        all_data = z["data"]    # (6209, 6, 32, 36)
        all_dates = z["dates"]  # (6209,)

        # Load static features
        static_file = artifacts_dir / "static_features.npy"
        if not static_file.is_file():
            raise FileNotFoundError(f"Static features not found: {static_file}")
        static = np.load(static_file).astype(np.float32)  # (2, 32, 36)

        # Normalize static features:
        # channel 0: land_sea_mask is already [0, 1]
        # channel 1: orography elevation in km (divide by 1000m for unit scale O(1))
        static_norm = static.copy()
        static_norm[1] = static_norm[1] / 1000.0
        self.static_features = torch.from_numpy(static_norm)  # (2, 32, 36)

        # Slice data for the requested split
        if split in SPLIT_RANGES:
            start_date, end_date = SPLIT_RANGES[split]
            mask = (all_dates >= start_date) & (all_dates <= end_date)
            split_data = all_data[mask]
            split_dates = all_dates[mask]
        else:
            split_data = all_data
            split_dates = all_dates

        # Z-score normalize dynamic data
        self.norm_data = self.normalizer.normalize(split_data, channel_axis=1)  # (N, 6, 32, 36)
        self.dates = split_dates

        # Precompute day-of-year embeddings [sin, cos] for all dates in split
        self.doy_sincos = np.array([doy_sincos(_date_to_doy(d)) for d in self.dates], dtype=np.float32)  # (N, 2)

        # Valid sample count: need t_in input days + step_ahead target day
        self.window_len = self.t_in + self.step_ahead
        self.n_samples = max(0, len(self.dates) - self.window_len + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch sample (x, y).

        x: (22, 32, 36) float32 tensor
        y: (6, 32, 36) float32 tensor
        """
        # 1. Dynamic history: t_in consecutive days -> (t_in, 6, 32, 36)
        hist = self.norm_data[idx : idx + self.t_in]  # (3, 6, 32, 36)
        # Flatten time into channel dimension -> (18, 32, 36)
        hist_flat = hist.reshape(self.t_in * NC, NLAT, NLON)
        hist_tensor = torch.from_numpy(hist_flat)

        # 2. DOY embedding for the LAST input day (idx + t_in - 1)
        last_input_idx = idx + self.t_in - 1
        doy_pair = self.doy_sincos[last_input_idx]  # [sin, cos]
        # Broadcast spatially to (2, 32, 36)
        doy_spatial = torch.from_numpy(doy_pair).view(2, 1, 1).expand(2, NLAT, NLON)

        # 3. Concatenate all 22 channels: [18 dynamic, 2 static, 2 doy]
        x = torch.cat([hist_tensor, self.static_features, doy_spatial], dim=0)  # (22, 32, 36)

        # 4. Target: day after last input day (idx + t_in) -> (6, 32, 36)
        target_idx = idx + self.t_in
        target = self.norm_data[target_idx]  # (6, 32, 36)
        y = torch.from_numpy(target)

        return x, y

    def get_sample_metadata(self, idx: int) -> dict:
        """Return date metadata for sample idx."""
        input_dates = self.dates[idx : idx + self.t_in].tolist()
        target_date = int(self.dates[idx + self.t_in])
        return {
            "input_dates": input_dates,
            "last_input_date": input_dates[-1],
            "target_date": target_date,
            "doy_sincos": self.doy_sincos[idx + self.t_in - 1].tolist(),
        }


def get_dataloader(
    split: Literal["train", "val", "test"] = "train",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    cache_path: str | Path = "data/cache/climate_data.npz",
    artifacts_dir: str | Path = "artifacts",
) -> DataLoader:
    """Convenience factory to create DataLoader for specified split."""
    dataset = ClimateSequenceDataset(
        split=split,
        cache_path=cache_path,
        artifacts_dir=artifacts_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
