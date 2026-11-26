# PHASE 2C COMPLETION REPORT
## Multiverse of Monsoon — Track 2: Earth-System Modelling
**Personal Project — AI-Powered Climate Digital Twin of India**

---

### 1. Executive Summary

Phase 2C (Data Extraction, Normalization & PyTorch Dataset Pipeline) is complete. The climate data from `india_climate_train.nc` has been extracted, audited, normalized according to official competition statistics, and integrated into a high-performance PyTorch dataset pipeline. All 21 unit tests across three test modules have executed and passed with 100% success.

---

### 2. Deliverables and Created Modules

| File Path | Description | Key Properties / Dimensions |
| :--- | :--- | :--- |
| `scripts/extract_data.py` | Standalone extraction and pre-computation script | Slices dynamic/static data, cleans coastal artifacts, computes 366-day climatology |
| `artifacts/static_features.npy` | Static geographical feature array | Shape `(2, 32, 36)`, `float32`, non-negative, 9,344 bytes |
| `artifacts/climatology.npy` | Official 366-day circular daily climatology | Shape `(366, 6, 32, 36)`, `float32`, 10,119,296 bytes ($\le 400$ limit) |
| `data/cache/climate_data.npz` | Local cached dynamic arrays & dates | Shape `(6209, 6, 32, 36)`, gitignored in `data/cache/` |
| `src/normalization.py` | Normalization & denormalization class | Handles NumPy & PyTorch tensors, clamps physical bounds (`tp >= 0`, `t2m` in `[233.15, 333.15]`) |
| `src/dataset.py` | PyTorch `Dataset` and `DataLoader` pipeline | Generates $(22, 32, 36)$ input tensor $\to (6, 32, 36)$ target tensor |
| `tests/test_data.py` | Unit tests for raw NetCDF data integrity | 8 tests verifying coordinates, channels, time continuity, and absence of NaNs |
| `tests/test_preprocessing.py` | Unit tests for statistics, normalization, artifacts | 7 tests verifying roundtrip consistency, physical clamping, and artifact validation |
| `tests/test_dataset.py` | Unit tests for PyTorch sequence dataset & splits | 6 tests verifying sample shapes, split boundary isolation, DOY broadcasting, and collation |

---

### 3. Chronological Splits & Boundary Isolation

Strict temporal boundaries have been enforced to prevent data leakage across train, validation, and test splits:

| Split | Date Range (Inclusive) | Total Days | Sequence Window ($t_{\text{in}} + 1$) | Usable Sequence Samples | Boundary Isolation Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2005-01-01 to 2018-12-31 | 5,113 days | 4 days ($3 + 1$) | **5,110 samples** | Last target date: 2018-12-31 |
| **Validation** | 2019-01-01 to 2020-12-31 | 731 days | 4 days ($3 + 1$) | **728 samples** | First input: 2019-01-01, Last target: 2020-12-31 |
| **Test** | 2021-01-01 to 2021-12-31 | 365 days | 4 days ($3 + 1$) | **362 samples** | First input: 2021-01-01, Last target: 2021-12-31 |

- **Zero Boundary Crossing**: No window spans across split boundaries.
- **Input Tensor Channels ($22 \times 32 \times 36$)**:
  - Channels $0-17$: 3 consecutive historical days $\times 6$ z-score normalized dynamic atmospheric variables.
  - Channels $18-19$: Static features (`land_sea_mask` in $[0, 1]$, `orography_m` scaled to km).
  - Channels $20-21$: Day-of-Year cyclical embedding $[\sin(2\pi \cdot \text{DOY}/365), \cos(2\pi \cdot \text{DOY}/365)]$ spatially broadcast across the entire $32 \times 36$ grid.
- **Target Tensor Channels ($6 \times 32 \times 36$)**:
  - Channels $0-5$: Dynamic atmospheric state on Day $t+1$ (z-score normalized).

---

### 4. Unit Test Verification Results

All tests executed via `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`:

```text
test_coordinates_alignment (test_data.TestDatasetIntegrity.test_coordinates_alignment) ... ok
test_dimensions (test_data.TestDatasetIntegrity.test_dimensions) ... ok
test_dynamic_channels_present (test_data.TestDatasetIntegrity.test_dynamic_channels_present) ... ok
test_file_exists_and_size (test_data.TestDatasetIntegrity.test_file_exists_and_size) ... ok
test_no_nans_or_infs (test_data.TestDatasetIntegrity.test_no_nans_or_infs) ... ok
test_precipitation_non_negative (test_data.TestDatasetIntegrity.test_precipitation_non_negative) ... ok
test_static_channels_present (test_data.TestDatasetIntegrity.test_static_channels_present) ... ok
test_time_continuity (test_data.TestDatasetIntegrity.test_time_continuity) ... ok
test_dataloader_batch_collation (test_dataset.TestClimateSequenceDataset.test_dataloader_batch_collation) ... ok
test_dataset_splits_and_lengths (test_dataset.TestClimateSequenceDataset.test_dataset_splits_and_lengths) ... ok
test_doy_sincos_broadcasting (test_dataset.TestClimateSequenceDataset.test_doy_sincos_broadcasting) ... ok
test_sample_shapes_and_types (test_dataset.TestClimateSequenceDataset.test_sample_shapes_and_types) ... ok
test_split_boundary_isolation (test_dataset.TestClimateSequenceDataset.test_split_boundary_isolation) ... ok
test_static_feature_scaling (test_dataset.TestClimateSequenceDataset.test_static_feature_scaling) ... ok
test_climatology_artifact (test_preprocessing.TestArtifactsIntegrity.test_climatology_artifact) ... ok
test_static_features_artifact (test_preprocessing.TestArtifactsIntegrity.test_static_features_artifact) ... ok
test_load_stats (test_preprocessing.TestNormalizationAndStats.test_load_stats) ... ok
test_numpy_normalization_roundtrip (test_preprocessing.TestNormalizationAndStats.test_numpy_normalization_roundtrip) ... ok
test_physical_clamping (test_preprocessing.TestNormalizationAndStats.test_physical_clamping) ... ok
test_stats_json_structure (test_preprocessing.TestNormalizationAndStats.test_stats_json_structure) ... ok
test_torch_normalization_roundtrip (test_preprocessing.TestNormalizationAndStats.test_torch_normalization_roundtrip) ... ok

----------------------------------------------------------------------
Ran 21 tests in 10.875s

OK
```

---

### 5. Architectural and Anti-Cheat Compliance

1. **No Database / ORM**: No relational database, SQLite, MongoDB, Prisma, or migration files exist or were introduced.
2. **No Pretrained Models / LLMs**: No external foundation models, weights, or tokenizers exist.
3. **Array File Restrictions**:
   - `artifacts/climatology.npy` holds exactly 366 days ($366 \le 400$, compliant).
   - `artifacts/static_features.npy` holds 2 channels ($2 \le 400$, compliant).
   - Raw multi-year extraction cache is located in `data/cache/climate_data.npz` which is excluded via `.gitignore`.
4. **Git Hygiene**: No commits, staging, branch switching, or remote pushes have been made.
