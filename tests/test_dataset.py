"""Unit tests for ClimateSequenceDataset and DataLoader."""
import unittest
from pathlib import Path
import numpy as np
import torch

from monsoon_r1.contract import NC, NLAT, NLON
from src.dataset import ClimateSequenceDataset, get_dataloader, SPLIT_RANGES

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "data" / "cache" / "climate_data.npz"
ARTIFACTS_DIR = ROOT / "artifacts"


class TestClimateSequenceDataset(unittest.TestCase):
    """Test PyTorch dataset and dataloader mechanics and split boundaries."""

    @classmethod
    def setUpClass(cls):
        if not CACHE_FILE.is_file():
            raise FileNotFoundError(f"Cache file missing: {CACHE_FILE}. Run scripts/extract_data.py first.")

    def test_dataset_splits_and_lengths(self):
        """Test dataset instantiation and sample counts for train, val, and test splits."""
        train_ds = ClimateSequenceDataset(split="train", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)
        val_ds = ClimateSequenceDataset(split="val", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)
        test_ds = ClimateSequenceDataset(split="test", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)

        # 2005-2018 (5113 days) -> 5110 samples
        self.assertEqual(len(train_ds), 5110)
        # 2019-2020 (731 days) -> 728 samples
        self.assertEqual(len(val_ds), 728)
        # 2021 (365 days) -> 362 samples
        self.assertEqual(len(test_ds), 362)

    def test_sample_shapes_and_types(self):
        """Test single sample tensor shapes, dtypes, and finite values."""
        ds = ClimateSequenceDataset(split="train", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)
        x, y = ds[0]

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

        # Expected: x is 22 channels (18 dynamic + 2 static + 2 doy), y is 6 dynamic channels
        self.assertEqual(x.shape, (22, NLAT, NLON))
        self.assertEqual(y.shape, (NC, NLAT, NLON))

        # Check absence of NaNs and Infs
        self.assertFalse(torch.isnan(x).any(), "Input tensor x contains NaNs.")
        self.assertFalse(torch.isinf(x).any(), "Input tensor x contains Infs.")
        self.assertFalse(torch.isnan(y).any(), "Target tensor y contains NaNs.")
        self.assertFalse(torch.isinf(y).any(), "Target tensor y contains Infs.")

    def test_split_boundary_isolation(self):
        """Verify strict chronological boundary isolation between train, val, and test."""
        train_ds = ClimateSequenceDataset(split="train", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)
        val_ds = ClimateSequenceDataset(split="val", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)
        test_ds = ClimateSequenceDataset(split="test", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)

        # Train: first input day is 2005-01-01, last target day is 2018-12-31
        meta_first_train = train_ds.get_sample_metadata(0)
        meta_last_train = train_ds.get_sample_metadata(len(train_ds) - 1)
        self.assertEqual(meta_first_train["input_dates"][0], 20050101)
        self.assertEqual(meta_last_train["target_date"], 20181231)
        self.assertLessEqual(meta_last_train["target_date"], 20181231)

        # Val: first input day is 2019-01-01, last target day is 2020-12-31
        meta_first_val = val_ds.get_sample_metadata(0)
        meta_last_val = val_ds.get_sample_metadata(len(val_ds) - 1)
        self.assertEqual(meta_first_val["input_dates"][0], 20190101)
        self.assertEqual(meta_last_val["target_date"], 20201231)
        self.assertGreaterEqual(meta_first_val["input_dates"][0], 20190101)
        self.assertLessEqual(meta_last_val["target_date"], 20201231)

        # Test: first input day is 2021-01-01, last target day is 2021-12-31
        meta_first_test = test_ds.get_sample_metadata(0)
        meta_last_test = test_ds.get_sample_metadata(len(test_ds) - 1)
        self.assertEqual(meta_first_test["input_dates"][0], 20210101)
        self.assertEqual(meta_last_test["target_date"], 20211231)
        self.assertGreaterEqual(meta_first_test["input_dates"][0], 20210101)
        self.assertLessEqual(meta_last_test["target_date"], 20211231)

    def test_doy_sincos_broadcasting(self):
        """Verify DOY sin and cos channels are spatially uniform and lie on the unit circle."""
        ds = ClimateSequenceDataset(split="train", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)
        x, _ = ds[100]

        doy_sin = x[20].numpy()
        doy_cos = x[21].numpy()

        # Spatially uniform: min == max across grid
        self.assertAlmostEqual(float(doy_sin.min()), float(doy_sin.max()), places=5)
        self.assertAlmostEqual(float(doy_cos.min()), float(doy_cos.max()), places=5)

        # Lies on unit circle: sin^2 + cos^2 == 1.0
        val_sin = float(doy_sin[0, 0])
        val_cos = float(doy_cos[0, 0])
        self.assertAlmostEqual(val_sin**2 + val_cos**2, 1.0, places=4)

    def test_static_feature_scaling(self):
        """Verify static features: land_sea_mask in [0, 1] and orography scaled to km (< 10.0)."""
        ds = ClimateSequenceDataset(split="train", cache_path=CACHE_FILE, artifacts_dir=ARTIFACTS_DIR)
        x, _ = ds[0]

        lsm = x[18].numpy()
        oro_km = x[19].numpy()

        self.assertTrue(np.all(lsm >= 0.0) and np.all(lsm <= 1.0))
        self.assertTrue(np.all(oro_km >= 0.0), "Orography contains negative values.")
        # Max elevation on Earth < 9 km (Mount Everest is 8.848 km)
        self.assertLess(float(oro_km.max()), 9.0)
        self.assertGreater(float(oro_km.max()), 5.0)

    def test_dataloader_batch_collation(self):
        """Test DataLoader batches correctly with shape (B, 22, 32, 36) and (B, 6, 32, 36)."""
        loader = get_dataloader(
            split="val",
            batch_size=16,
            shuffle=False,
            num_workers=0,
            cache_path=CACHE_FILE,
            artifacts_dir=ARTIFACTS_DIR,
        )

        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.shape, (16, 22, NLAT, NLON))
        self.assertEqual(batch_y.shape, (16, NC, NLAT, NLON))
        self.assertEqual(batch_x.dtype, torch.float32)
        self.assertEqual(batch_y.dtype, torch.float32)
        self.assertFalse(torch.isnan(batch_x).any())
        self.assertFalse(torch.isnan(batch_y).any())


if __name__ == "__main__":
    unittest.main()
