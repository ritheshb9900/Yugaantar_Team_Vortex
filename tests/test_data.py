"""Unit tests for climate dataset integrity, dimensions, coordinates, and physical properties."""
import unittest
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON, LAT, LON

ROOT = Path(__file__).resolve().parent.parent
NC_PATH = ROOT / "india_climate_train.nc"


class TestDatasetIntegrity(unittest.TestCase):
    """Verify raw NetCDF dataset exists, conforms to contract, and is clean."""

    @classmethod
    def setUpClass(cls):
        if not NC_PATH.is_file():
            raise FileNotFoundError(f"Missing required dataset: {NC_PATH}")
        cls.ds = nc.Dataset(NC_PATH, "r")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "ds") and cls.ds.isopen():
            cls.ds.close()

    def test_file_exists_and_size(self):
        """Test dataset file existence and byte size (~122.8 MB)."""
        self.assertTrue(NC_PATH.is_file(), "india_climate_train.nc does not exist in root.")
        size_bytes = NC_PATH.stat().st_size
        self.assertEqual(
            size_bytes,
            122831709,
            f"File size mismatch: expected 122,831,709 bytes, got {size_bytes}",
        )

    def test_dimensions(self):
        """Test spatial, temporal, and channel dimensions."""
        self.assertIn("time", self.ds.dimensions)
        self.assertIn("channel", self.ds.dimensions)
        self.assertIn("lat", self.ds.dimensions)
        self.assertIn("lon", self.ds.dimensions)
        self.assertIn("static_channel", self.ds.dimensions)

        self.assertEqual(len(self.ds.dimensions["time"]), 6209)
        self.assertEqual(len(self.ds.dimensions["channel"]), NC)
        self.assertEqual(len(self.ds.dimensions["lat"]), NLAT)
        self.assertEqual(len(self.ds.dimensions["lon"]), NLON)
        self.assertEqual(len(self.ds.dimensions["static_channel"]), 2)

    def test_coordinates_alignment(self):
        """Test latitude and longitude coordinates match contract exactly."""
        lats = np.array(self.ds.variables["lat"][:], dtype=np.float32)
        lons = np.array(self.ds.variables["lon"][:], dtype=np.float32)

        self.assertEqual(len(lats), NLAT)
        self.assertEqual(len(lons), NLON)
        np.testing.assert_allclose(lats, LAT, atol=1e-4)
        np.testing.assert_allclose(lons, LON, atol=1e-4)

    def test_dynamic_channels_present(self):
        """Test all 6 dynamic channels exist in the NetCDF dataset in correct order."""
        channels_in_nc = [str(c) for c in self.ds.variables["channel"][:]]
        self.assertEqual(channels_in_nc, list(CHANNELS))

        data_var = self.ds.variables["data"]
        self.assertEqual(
            data_var.shape,
            (6209, NC, NLAT, NLON),
            f"Data array shape mismatch: expected (6209, {NC}, {NLAT}, {NLON}), got {data_var.shape}",
        )

    def test_static_channels_present(self):
        """Test static variables land_sea_mask and orography_m exist and match shape."""
        static_channels = [str(c) for c in self.ds.variables["static_channel"][:]]
        self.assertEqual(static_channels, ["land_sea_mask", "orography_m"])

        static_var = self.ds.variables["static"]
        self.assertEqual(static_var.shape, (2, NLAT, NLON))

        lsm = np.array(static_var[0, :, :])
        oro = np.array(static_var[1, :, :])

        # Check values
        self.assertTrue(np.all(lsm >= 0.0) and np.all(lsm <= 1.0))
        # Orography: max elevation in Himalayas should be > 5000 m
        self.assertGreater(float(np.max(oro)), 5000.0)

    def test_time_continuity(self):
        """Test time range is exactly 2005-01-01 to 2021-12-31 without gaps or duplicates."""
        time_var = self.ds.variables["time"]
        dates = nc.num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        expected_range = pd.date_range("2005-01-01", "2021-12-31", freq="D").strftime("%Y-%m-%d").tolist()
        self.assertEqual(len(date_strs), 6209)
        self.assertEqual(date_strs, expected_range)

    def test_no_nans_or_infs(self):
        """Test that data array has zero NaN or Inf values across all timesteps and channels."""
        data = np.array(self.ds.variables["data"][:])
        self.assertFalse(np.isnan(data).any(), "Data array contains NaN values.")
        self.assertFalse(np.isinf(data).any(), "Data array contains Inf values.")

        static = np.array(self.ds.variables["static"][:])
        self.assertFalse(np.isnan(static).any(), "Static array contains NaN values.")
        self.assertFalse(np.isinf(static).any(), "Static array contains Inf values.")

    def test_precipitation_non_negative(self):
        """Test that precipitation (channel index 1) is non-negative everywhere."""
        tp = np.array(self.ds.variables["data"][:, 1, :, :])
        self.assertTrue(np.all(tp >= 0.0), "Precipitation contains negative values.")


if __name__ == "__main__":
    unittest.main()
