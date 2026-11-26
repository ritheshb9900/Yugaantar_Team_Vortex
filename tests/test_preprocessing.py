"""Unit tests for statistics, channel normalization, and preprocessing artifacts."""
import json
import unittest
from pathlib import Path
import numpy as np
import torch

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON
from src.normalization import ChannelNormalizer, load_stats

ROOT = Path(__file__).resolve().parent.parent
STATS_FILE = ROOT / "stats.json"
CLIMATOLOGY_FILE = ROOT / "artifacts" / "climatology.npy"
STATIC_FEATURES_FILE = ROOT / "artifacts" / "static_features.npy"


class TestNormalizationAndStats(unittest.TestCase):
    """Test official normalization statistics and normalizer behavior."""

    def setUp(self):
        self.normalizer = ChannelNormalizer(STATS_FILE)

    def test_stats_json_structure(self):
        """Test stats.json has expected keys and correct channel count."""
        self.assertTrue(STATS_FILE.is_file(), "stats.json does not exist.")
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("channels", data)
        self.assertIn("mean", data)
        self.assertIn("std", data)

        self.assertEqual(tuple(data["channels"]), CHANNELS)
        self.assertEqual(len(data["mean"]), NC)
        self.assertEqual(len(data["std"]), NC)

        # Standard deviations must be strictly positive
        for s in data["std"]:
            self.assertGreater(s, 0.0)

    def test_load_stats(self):
        """Test load_stats function returns numpy arrays of correct shapes and values."""
        mean, std, channels = load_stats(STATS_FILE)
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(std, np.ndarray)
        self.assertEqual(mean.shape, (6,))
        self.assertEqual(std.shape, (6,))
        self.assertEqual(tuple(channels), CHANNELS)

    def test_numpy_normalization_roundtrip(self):
        """Test Z = (X - mu)/sigma and X = Z * sigma + mu round-trip for numpy arrays."""
        # Create synthetic physical weather array (6, 32, 36)
        x = np.zeros((NC, NLAT, NLON), dtype=np.float32)
        x[0] = 300.0   # t2m ~ 300K
        x[1] = 10.0    # tp ~ 10mm
        x[2] = 2.0     # u10 ~ 2m/s
        x[3] = -1.5    # v10 ~ -1.5m/s
        x[4] = 101000. # msl ~ 1010hPa
        x[5] = 45.0    # tcwv ~ 45kg/m2

        z = self.normalizer.normalize(x, channel_axis=0)
        self.assertEqual(z.shape, x.shape)
        self.assertEqual(z.dtype, np.float32)

        x_rec = self.normalizer.denormalize(z, channel_axis=0, clamp_physical=False)
        np.testing.assert_allclose(x_rec, x, atol=1e-5)

    def test_torch_normalization_roundtrip(self):
        """Test round-trip normalization for PyTorch tensors."""
        # Batch tensor (B, 6, 32, 36)
        batch_size = 4
        x = torch.zeros((batch_size, NC, NLAT, NLON), dtype=torch.float32)
        x[:, 0] = 295.0
        x[:, 1] = 5.0
        x[:, 2] = 1.0
        x[:, 3] = -2.0
        x[:, 4] = 100500.0
        x[:, 5] = 40.0

        z = self.normalizer.normalize(x, channel_axis=1)
        self.assertEqual(z.shape, x.shape)
        self.assertEqual(z.dtype, torch.float32)

        x_rec = self.normalizer.denormalize(z, channel_axis=1, clamp_physical=False)
        self.assertTrue(torch.allclose(x_rec, x, atol=1e-4))

    def test_physical_clamping(self):
        """Test physical clipping on denormalization: tp >= 0, t2m in [233.15, 333.15]."""
        # Create unphysically extreme z-scores
        z_np = np.zeros((NC, NLAT, NLON), dtype=np.float32)
        z_np[0] = -50.0  # Would give extremely low t2m (< 0 K)
        z_np[1] = -50.0  # Would give negative precipitation

        x_clamped = self.normalizer.denormalize(z_np, channel_axis=0, clamp_physical=True)
        self.assertAlmostEqual(float(x_clamped[0].min()), 233.15, places=3)
        self.assertGreaterEqual(float(x_clamped[1].min()), 0.0)

        # Torch version
        z_torch = torch.from_numpy(z_np).unsqueeze(0)  # (1, 6, 32, 36)
        x_torch_clamped = self.normalizer.denormalize(z_torch, channel_axis=1, clamp_physical=True)
        self.assertAlmostEqual(float(x_torch_clamped[:, 0].min()), 233.15, places=3)
        self.assertGreaterEqual(float(x_torch_clamped[:, 1].min()), 0.0)


class TestArtifactsIntegrity(unittest.TestCase):
    """Test saved artifacts static_features.npy and climatology.npy."""

    def test_climatology_artifact(self):
        """Test climatology.npy shape, absence of NaNs, and physical bounds."""
        self.assertTrue(CLIMATOLOGY_FILE.is_file(), "climatology.npy artifact does not exist.")
        clim = np.load(CLIMATOLOGY_FILE)

        self.assertEqual(clim.shape, (366, NC, NLAT, NLON))
        self.assertFalse(np.isnan(clim).any(), "Climatology contains NaNs.")
        self.assertFalse(np.isinf(clim).any(), "Climatology contains Infs.")

        # Physical sanity checks
        self.assertTrue(np.all(clim[:, 1] >= 0.0), "Climatology precipitation contains negative values.")
        self.assertTrue(np.all(clim[:, 0] > 230.0), "Climatology t2m has unphysically low values.")
        self.assertTrue(np.all(clim[:, 0] < 340.0), "Climatology t2m has unphysically high values.")

    def test_static_features_artifact(self):
        """Test static_features.npy shape, absence of NaNs, and non-negativity."""
        self.assertTrue(STATIC_FEATURES_FILE.is_file(), "static_features.npy artifact does not exist.")
        static = np.load(STATIC_FEATURES_FILE)

        self.assertEqual(static.shape, (2, NLAT, NLON))
        self.assertFalse(np.isnan(static).any(), "Static features contain NaNs.")
        self.assertFalse(np.isinf(static).any(), "Static features contain Infs.")

        # Channel 0: land-sea mask in [0, 1]
        lsm = static[0]
        self.assertTrue(np.all(lsm >= 0.0) and np.all(lsm <= 1.0))

        # Channel 1: orography elevation non-negative (coastal artifacts clamped)
        oro = static[1]
        self.assertTrue(np.all(oro >= 0.0), "Orography contains negative values after cleaning.")
        self.assertGreater(float(np.max(oro)), 5000.0, "Orography max elevation should exceed 5000m.")


if __name__ == "__main__":
    unittest.main()
