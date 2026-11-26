"""Extract, validate, and preprocess raw NetCDF climate data.

Outputs:
  - artifacts/static_features.npy: Shape (2, 32, 36) [land_sea_mask, orography_m]
  - artifacts/climatology.npy:     Shape (366, 6, 32, 36) 366-day circular smooth climatology
  - data/cache/climate_data.npz:   Extracted dynamic data (6209, 6, 32, 36) + dates (gitignored)
"""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON, LAT, LON
import monsoon_r1.stats as r1_stats


def extract_dataset(
    nc_path: Path,
    artifacts_dir: Path,
    cache_dir: Path,
) -> dict:
    """Extract and validate NetCDF data into artifacts and local cache."""
    nc_path = Path(nc_path).resolve()
    artifacts_dir = Path(artifacts_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening NetCDF dataset: {nc_path}")
    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF dataset not found: {nc_path}")

    # Open dataset read-only
    with xr.open_dataset(nc_path) as ds:
        # 1. Verify and extract dynamic channels in official order
        print("Extracting dynamic channels...")
        ds_channels = list(ds.coords["channel"].values)
        if ds_channels != list(CHANNELS):
            raise ValueError(f"Channel mismatch: {ds_channels} != {list(CHANNELS)}")

        data = ds["data"].values.astype("float32")  # Shape (6209, 6, 32, 36)
        if data.shape != (6209, NC, NLAT, NLON):
            raise ValueError(f"Unexpected dynamic shape: {data.shape}")

        # Check finite
        if not np.isfinite(data).all():
            raise ValueError("Dataset contains non-finite values (NaN/Inf)!")

        # Clamp tiny negative rain artifacts if any
        data[:, 1] = np.maximum(data[:, 1], 0.0)

        # 2. Extract and format date array
        time_vals = pd.to_datetime(ds.coords["time"].values)
        dates_int = np.array([int(d.strftime("%Y%m%d")) for d in time_vals], dtype=np.int32)
        if len(dates_int) != 6209:
            raise ValueError(f"Expected 6209 dates, got {len(dates_int)}")

        # Verify daily continuity
        diffs = np.diff(time_vals.values).astype("timedelta64[D]").astype(int)
        if not (diffs == 1).all():
            raise ValueError("Dates are not continuous daily intervals!")

        # 3. Extract and sanitize static features
        print("Extracting static features...")
        lsm = ds["static"].sel(static_channel="land_sea_mask").values.astype("float32")
        oro = ds["static"].sel(static_channel="orography_m").values.astype("float32")

        # Sanitize static features
        lsm = np.clip(lsm, 0.0, 1.0)
        oro = np.maximum(oro, 0.0)  # Clamp coastal sea-level negative artifacts (-0.16m)

        static_features = np.stack([lsm, oro], axis=0).astype("float32")  # Shape (2, 32, 36)
        if static_features.shape != (2, NLAT, NLON):
            raise ValueError(f"Unexpected static shape: {static_features.shape}")

        # 4. Save static features artifact
        static_out = artifacts_dir / "static_features.npy"
        np.save(static_out, static_features)
        print(f"Saved static features: {static_out} (shape: {static_features.shape}, size: {static_out.stat().st_size} bytes)")

        # 5. Compute and save climatology artifact
        print("Computing 366-day smoothed climatology...")
        clim = r1_stats.climatology(data, dates_int, smooth_days=15).astype("float32")
        if clim.shape != (366, NC, NLAT, NLON):
            raise ValueError(f"Unexpected climatology shape: {clim.shape}")
        if not np.isfinite(clim).all():
            raise ValueError("Climatology contains non-finite values!")

        clim_out = artifacts_dir / "climatology.npy"
        np.save(clim_out, clim)
        print(f"Saved climatology: {clim_out} (shape: {clim.shape}, size: {clim_out.stat().st_size} bytes)")

        # 6. Save extracted dynamic data to local cache outside Git tracking
        cache_out = cache_dir / "climate_data.npz"
        np.savez_compressed(
            cache_out,
            data=data,
            dates=dates_int,
            channels=np.array(list(CHANNELS)),
            lat=LAT,
            lon=LON,
        )
        print(f"Saved cached climate data: {cache_out} (size: {cache_out.stat().st_size / (1024*1024):.2f} MB)")

    return {
        "dynamic_shape": list(data.shape),
        "static_shape": list(static_features.shape),
        "climatology_shape": list(clim.shape),
        "start_date": int(dates_int[0]),
        "end_date": int(dates_int[-1]),
        "total_days": len(dates_int),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract climate dataset into artifacts and local cache.")
    parser.add_argument("--nc", default="india_climate_train.nc", help="Path to raw NetCDF file")
    parser.add_argument("--artifacts", default="artifacts", help="Directory for public artifacts")
    parser.add_argument("--cache", default="data/cache", help="Directory for local training cache")
    args = parser.parse_args()

    result = extract_dataset(Path(args.nc), Path(args.artifacts), Path(args.cache))
    print("\nExtraction Summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
