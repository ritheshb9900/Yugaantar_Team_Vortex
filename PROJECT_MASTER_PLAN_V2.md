# Multiverse of Monsoon: Project Master Plan (Version 2.0 — Deep Learning Only)
## Track 2: Earth-System Modelling – AI-Powered Climate Digital Twin of India

---

### Executive Statement: Deep Learning Weather Engine Only (No LLMs)
This document is the authoritative master architecture blueprint for **Track 2: Multiverse of Monsoon**.

**Architectural Decision**: This project builds and trains an autonomous, numerical **Deep Learning Climate Digital Twin from scratch using PyTorch**. 
- **NO Large Language Models (LLMs)**, GPTs, tokenizers, text datasets, Transformer language models, or chatbots are part of this project.
- **NO pretrained external weather models** (e.g., GraphCast, Pangu-Weather, ClimaX, FourCastNet) or downloaded weights are used.
- All modeling, data engineering, optimization, offline serving, and hazard risk assessments are performed via deterministic physical tensor operations and deep neural networks trained strictly from randomly initialized weights on the provided 2005–2021 historical climate dataset.

---

## 1. Authoritative Compliance Matrix & Verification Status

> [!NOTE]
> **Status Clarification**: In accordance with strict planning governance, all execution benchmarks and performance scores below are classified as **PLANNED / NOT RUN**. Actual numerical performance and scores will be established through real implementation and empirical testing during subsequent phases.

| Requirement Domain | 1. Official Mandatory (Automated Grader) | 2. Presentation & Desk-Review | 3. Optional Internal Targets | 4. Current Status in Workspace |
|---|---|---|---|---|
| **Input / Output Wire Format** | Plain nested JSON lists of finite `float32`. Input: `(3, 6, 32, 36)`. Output: `(steps, 6, 32, 36)`. Strict physical units. | Live dashboard issuing HTTP POST `/simulate` requests and displaying dynamic JSON response. | Base64 compact encoding (`{"b64":..., "shape":[...]}`) supported by client. | **PLANNED / NOT RUN** (Specification locked in API Contract; implementation in Phase 7). |
| **Model Provenance & Weights** | **Trained strictly from scratch** on supplied 2005–2021 ERA5 data. Checkpoint inside repo ($< 2.0\text{ GB}$). | Defense of architectural choices, loss convergence, and gradient flow during desk review. | Internal target: model checkpoint size $< 50\text{ MB}$ ($\approx 5\text{ MB}$ planned). | **PLANNED / NOT RUN** (Random initialization; training in Phase 5; no weights exist yet). |
| **Execution Environment** | **100% Offline**, 4 CPU cores, 6 GB RAM, zero GPU, zero internet. Time limit: **$< 5.0\text{ s}$ per 10-step request**. | Interactive web dashboard demonstrating live model rollouts and shock simulations. | Internal experimental target: single-step forward pass $\approx 15\text{ ms}$, rollout $< 500\text{ ms}$. | **PLANNED / NOT RUN** (Host verified: 4 cores, offline container targeted; verification in Phase 7). |
| **Forecasting Skill** | Evaluated at Lead Days 1, 4, 8 across 5 hidden 2023–2024 windows. **Must beat Day-4 persistence** (Hard qualification gate). | Explanation of how forecast drift and blur are prevented over 10 days. | Ensemble rollout for uncertainty estimation. | **PLANNED / NOT RUN** (Target benchmark to be empirically measured against persistence in Phase 5). |
| **Dynamics & Anti-Blur** | Day-8 spatial variance ratio $\in [0.40, 2.50]$ per channel. (10 Pts). | Spatial gradient comparison and energy spectrum preservation. | High-frequency Fourier loss penalization. | **PLANNED / NOT RUN** (Target benchmark to be evaluated via `monsoon_r1.scoring` in Phase 6). |
| **Counterfactual Shocks** | 3 Official Shocks: (1) Warm BoB (+3K, +15% moisture), (2) Monsoon Stall (-40% winds), (3) Null Shock ($\Delta=0$, tolerance $< 10^{-3}\sigma$). (15 Pts). | Additional presentation scenario: $+2.5^\circ\text{C}$ Arabian Sea Heat Anomaly (`t2m`). Live judge shock test. | Interactive drag-and-drop bounding box shock creator on web UI. | **PLANNED / NOT RUN** (Physics masks defined; causality harness to be executed in Phase 6). |
| **Real-World Impact** | Not scored by automated judge. | **Mandatory for Stage 2 Desk Review & Technical Brief Page 2**. Must be calculated LIVE from model outputs. | Multi-tier hazard alert dispatch simulations. | **PLANNED / NOT RUN** (Mathematical formulas specified; rule-based engine to be built in Phase 8). |
| **Repository Integrity** | Scored via `repo_scan.py` (15 Pts). No external weather APIs, no hardcoded dates, no data archives $> 400$ days. | Clean Git commit history, clear modular architecture. | Comprehensive unit test suite in `tests/`. | **PLANNED / NOT RUN** (Static rules audited; full pass to be verified in Phase 1 & 7). |
| **Submission Artifacts** | `serve.py` at root, trained weights, separate UI code, comprehensive `README.md`. | Two-page Technical Brief PDF (Page 1: ML Model & Drift; Page 2: Dashboard & Impact Extension). | Video walkthrough / demo recording. | **PLANNED / NOT RUN** (Deliverables assigned in Team Task Distribution). |

---

## 2. Dataset Hygiene & Repo Integrity (Anti-Cheat Compliance)

### 2.1 Strict Prohibition on Raw/Processed Multi-Year Archives in Repository
The organizer's static scanner (`monsoon_r1.repo_scan.py`) explicitly flags any array file in the repository with more than 400 daily timesteps:
```python
# monsoon_r1/repo_scan.py:
if s and len(s) >= 3 and s[0] > 400 and tuple(s[-2:]) == (32, 36):
  hits.append(f'{p.relative_to(root)} shape {tuple(s)}')
  # Triggers: flags.append({"level": "critical", "what": f"multi-year daily archive in the repo: {h}"})
```
- **Policy**: Under no circumstances will a multi-year daily archive (such as `data_train.npy` or `india_climate_train.nc`) be saved inside the submitted repository.
- **Handling Strategy**: 
  - The raw historical file `india_climate_train.nc` remains in `.gitignore`.
  - During Phase 2, local preprocessing generates intermediate training arrays in a local cache directory outside the Git tracking index (or in `data/` explicitly gitignored).
  - The only data arrays committed to `artifacts/` are small, compliant support files:
    1. `static_features.npy` ($2 \times 32 \times 36 \approx 9.2\text{ KB}$, permanent topography).
    2. `climatology.npy` ($366 \times 6 \times 32 \times 36 \approx 10.1\text{ MB}$, one-year climatology $\le 400$ days, explicitly permitted by `repo_scan.py`).
    3. `stats.json` (355 bytes, normalization constants).
    4. `model_weights.pt` ($\approx 5\text{ MB}$, model state dict).

### 2.2 Climatology & Leap-Year Handling Verification
The 366-day climatology strictly follows the official toolkit convention (`monsoon_r1.stats.climatology`):
- Calendar year 2000 is used as the reference leap year, mapping calendar dates to indices $0 \dots 365$.
- Feb 28 is day 58 (0-indexed: index 58), Feb 29 is index 59, and March 1 is index 60.
- A 15-day circular running mean (`smooth_days=15`) is applied across calendar days, ensuring smooth continuity across year boundaries (Dec 31 $\to$ Jan 1) and robust values for Feb 29.

---

## 3. End-to-End Normalization & Tensor Pipeline

To guarantee numerical consistency and avoid scale mismatches across heterogeneous atmospheric units, all operations are rigorously defined across normalized and physical spaces:

```
[Wire Input: Physical Units]
  init_states: (3, 6, 32, 36)
  perturbation: (6, 32, 36) or None
           │
           ▼
[Step 1: Perturbation Injection in Physical Space]
  X*(t) = init_states[-1] + perturbation (if present)
           │
           ▼
[Step 2: Z-Score Normalization via stats.json]
  Z(t_i) = (X(t_i) - μ) / σ   for i in {-2, -1, 0}
  (Dimensionless z-scores, zero-mean, unit-variance)
           │
           ▼
[Step 3: Feature Concatenation]
  Concat [Z(t-2), Z(t-1), Z*(t)]            -> (18, 32, 36)
  Concat static_features [lsm, oro_norm]     -> (2, 32, 36)
  Concat broadcast doy_sincos(t)             -> (2, 32, 36)
  ──────────────────────────────────────────────────────────
  Total Input to Res-UNet                    -> (Batch, 22, 32, 36)
           │
           ▼
[Step 4: Neural Forward Pass (Normalized Residual Tendency)]
  ΔZ = Model(Input)                         -> (Batch, 6, 32, 36)
  Z_hat(t+1) = Z*(t) + ΔZ                   (Normalized Next-Day Prediction)
           │
           ▼
[Step 5: Autoregressive Feedback Loop (In Normalized Space)]
  Update context buffer: [Z(t-1), Z*(t), Z_hat(t+1)]
  Advance DOY: doy_sincos(t+1)
  Repeat Steps 3 to 5 for k = 1 to 10
  (Zero intermediate denormalization roundtrips: prevents floating-point drift)
           │
           ▼
[Step 6: Denormalization to Physical Units at Exit]
  X_hat(t+k) = Z_hat(t+k) * σ + μ
  Clamp physical bounds:
    tp >= 0.0 mm/day
    t2m in [233.15, 333.15] K ([-40°C, +60°C])
           │
           ▼
[Wire Output: Physical Units]
  rollout: (10, 6, 32, 36) nested JSON list
```

---

## 4. Dimensionless Latitude-Weighted Training Loss

### 4.1 Physical Units Disparity & Official Alignment
In raw physical units, channel scales vary by five orders of magnitude:
- `msl` (pressure) $\approx 101,122\text{ Pa}$ ($\sigma \approx 619\text{ Pa}$)
- `t2m` (temperature) $\approx 292\text{ K}$ ($\sigma \approx 13.5\text{ K}$)
- `tp` (precipitation) $\approx 3.6\text{ mm}$ ($\sigma \approx 7.5\text{ mm}$)
- `u10`, `v10` (winds) $\approx 0.5\text{ m/s}$ ($\sigma \approx 2.5\text{ m/s}$)

If an unweighted raw mean squared error were used for training, pressure gradients would dominate by $10,000\times$, completely starving rainfall and wind gradients. 

In the official scoring harness (`monsoon_r1.scoring.py`), skill scores are calculated **per channel** and then averaged equally. Therefore, during **training**, our primary loss function computes latitude-weighted RMSE on **normalized dimensionless states**:

$$\mathcal{L}_{\text{primary}} = \frac{1}{6} \sum_{c=1}^6 \sqrt{ \frac{1}{\sum_{i=1}^{32} w_i} \sum_{i=1}^{32}\sum_{j=1}^{36} w_i \left( \hat{Z}_{c,i,j} - Z_{c,i,j} \right)^2 }$$

where:
$$w_i = \frac{\cos(\phi_i)}{\frac{1}{32}\sum_{k=1}^{32}\cos(\phi_k)}$$

This guarantees that each of the six atmospheric variables contributes equally to parameter updates, perfectly aligning the optimization objective with the competition scoring metric.

### 4.2 Candidate Experimental Auxiliary Losses
During Phase 4 and Phase 5, we will evaluate whether adding the following terms improves validation scores:
1. **Normalized MAE ($\mathcal{L}_{\text{MAE}}$)**: Linear penalty promoting robustness against non-Gaussian distribution tails.
2. **Heavy-Rain CSI Penalty ($\mathcal{L}_{\text{rain-CSI}}$)**: Extra gradient weighting on grid cells where observed rainfall $\text{tp} \ge 10\text{ mm/day}$ ($\approx Z_{\text{tp}} \ge 0.84$).
3. **Sobolev Spatial Gradient Loss ($\mathcal{L}_{\text{sobolev}}$)**: Penalizes blurring of spatial fronts, tested to maintain Day-8 spatial variance ratio $\ge 0.40$.

---

## 5. Scientifically Defensible Real-World Impact Extension

### 5.1 Rejection of False Near-Surface Humidity from Column Moisture
Atmospheric physics establishes that `tcwv` is **Total Column Water Vapour** ($\text{kg/m}^2$), integrating water vapor over the entire vertical column of the atmosphere. It cannot be mathematically inverted into near-surface 2-metre relative humidity ($RH_{2\text{m}}$) without vertical temperature and moisture soundings. Claiming to compute standard near-surface Heat Index directly from `tcwv` is scientifically unsupportable.

### 5.2 Defensible Temperature-Stress & Hazard Risk Indicators
Our real-world impact module (`src/impact.py`) computes transparent, live indicators derived from the predicted physical channels:

1. **IMD Heatwave & Climatological Departure Index**:
   - Near-surface air temperature converted to Celsius: $T^\circ\text{C} = \text{t2m} - 273.15$.
   - Positive departure above the 17-year smoothed daily climatology: $\Delta T = T^\circ\text{C} - T_{\text{clim}}^\circ\text{C}(\text{doy})$.
   - Classification adhering to official **India Meteorological Department (IMD)** criteria:
     - *Normal*: $T^\circ\text{C} < 40^\circ\text{C}$.
     - *Heatwave Alert*: $T^\circ\text{C} \ge 40^\circ\text{C}$ (plains) with $\Delta T \ge +4.5^\circ\text{C}$.
     - *Severe Heatwave Alert*: $T^\circ\text{C} \ge 45^\circ\text{C}$ or $\Delta T \ge +6.4^\circ\text{C}$.
2. **Cumulative Thermal Stress (Degree-Days)**:
   - Daily thermal excess above critical biological baseline: $\text{CDD}_{35} = \max(0, T^\circ\text{C} - 35^\circ\text{C})$, accumulated over consecutive days to identify livestock and crop heat stress.
3. **Extreme Rainfall & Flash Flood Hazard Index (FRI)**:
   - Daily rainfall evaluated against standard IMD precipitation classes:
     - *Heavy Rain*: $64.5 - 115.5\text{ mm/day}$
     - *Very Heavy Rain*: $115.6 - 204.4\text{ mm/day}$
     - *Extremely Heavy Rain*: $\ge 204.5\text{ mm/day}$
   - 3-day rolling cumulative rainfall $R_{\text{cum}} = \sum_{k=0}^2 \text{tp}(t-k)$ combined with terrain slope derived from static `orography_m` to highlight flash runoff corridors in the Western Ghats and Himalayan foothills.
4. **Column Moisture Availability Indicator**:
   - `tcwv` is displayed directly as column moisture loading ($\text{kg/m}^2$) to visualize monsoon moisture surges, without conflating it with surface relative humidity.
5. **Scientific Boundaries & Disclaimers**:
   - Indicators represent *meteorological hazard potential* at $1^\circ$ resolution, not official IMD civil warnings.
   - All indicators are computed **live** from model tensor outputs in real time. Zero pre-rendered maps exist.

---

## 6. Official Performance Limits vs Internal Targets

| Performance Dimension | Official Mandatory Limit (Grading Gate) | Internal Optimization Target (Engineering Goal) |
|---|---|---|
| **10-Step Inference Latency** | **$< 5.0\text{ seconds}$** total wall-clock time as measured by judge. Exceeding 5.0 s fails the window. | **$< 500\text{ ms}$** total rollout latency on 4 CPU cores ($> 10\times$ safety margin; single step target $\approx 15\text{ ms}$). |
| **Model Checkpoint Size** | **$< 2.0\text{ GB}$** maximum file size in repository. | **$< 50\text{ MB}$** checkpoint target (our planned Res-UNet is $\approx 5\text{ MB}$). |
| **Container RAM Budget** | **$< 6.0\text{ GB}$** total RAM consumption. | **$< 500\text{ MB}$** runtime RAM usage ($\approx 8\%$ of container limit). |
| **Day-4 Forecast Skill** | Must strictly outperform persistence baseline at Day 4. | Target skill score $\text{SS}_{\text{ref}} \ge 0.12$ (earns maximum 30 RMSE points). |
| **Heavy Rainfall CSI** | Evaluated at $10\text{ mm/day}$ threshold. | Target CSI skill score $\ge 0.10$ (earns maximum 15 CSI points). |
| **Day-8 Anti-Blur Ratio** | Spatial variance ratio must stay within $[0.40, 2.50]$. | Target variance ratio $\approx 0.70 - 1.10$. |
| **Causality Checks** | Finite, non-zero, localized Day 1 ($\ge 60\%$), outward propagation, stable $\le 5\sigma$, null consistency $< 10^{-3}\sigma$. | Target full $15/15$ causality score on official shocks. |

---

## 7. Submission Checklist & Hard Qualification Gates

All items below represent target criteria to be verified during execution:

- [ ] **Hard Gate 1 (Physical Validity)**: All 8 rollout requests return 10-step forecasts with finite values, zero negative rainfall ($\text{tp} \ge 0$), and temperature bounded within $[-40^\circ\text{C}, +60^\circ\text{C}]$. *(Target verification in Phase 6 & 7)*
- [ ] **Hard Gate 2 (Day-4 Persistence)**: Day-4 latitude-weighted RMSE beats persistence baseline across all channels. *(Target verification in Phase 5)*
- [ ] **Hard Gate 3 (Causality)**: Non-zero, localized response on Day 1 for both physical shocks; outward propagation $\le 8\text{ cells/day}$; stability $\le 5\sigma$; null shock deterministic within $10^{-3}\sigma$. *(Target verification in Phase 6)*
- [ ] **Hard Gate 4 (Offline Serving)**: `python serve.py --port 8000` starts offline with zero internet access; passes `python -m monsoon_r1.schema_validator http://localhost:8000` with 10-step latency $< 5.0\text{ s}$. *(Target verification in Phase 7)*
- [ ] **Hard Gate 5 (Repo Scan Score 15/15)**: `monsoon_r1.repo_scan.scan('.')` scores $8/8$ on anti-cheat and $7/7$ on code health (separate server, training pipeline present, weights inside repo, UI present, README present). *(Target verification in Phase 1 & 7)*
- [ ] **Hard Gate 6 (Weight Budget)**: Model checkpoint file (`artifacts/model_weights.pt`) is $< 2.0\text{ GB}$ (internal target $\approx 5\text{ MB}$). *(Target verification in Phase 5)*
- [ ] **Hard Gate 7 (Two-Page Technical Brief)**: Two-page PDF compiled and checked against required sections. *(Target verification in Phase 8)*
