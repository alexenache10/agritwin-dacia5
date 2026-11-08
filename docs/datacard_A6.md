
# Data Card: DACIA5 Benchmark (Problem 1)

**Mini-Twin:** AgriTwin (ID: A6) — Early Crop Identification in Romania

**Dataset Reference:** DACIA5: A Sentinel-1 and Sentinel-2 dataset for agricultural crop identification applications (*Big Earth Data*, 2025, DOI: [10.1080/20964471.2025.2512685](https://doi.org/10.1080/20964471.2025.2512685))

**Repository:** [https://github.com/alexenache10/agritwin-dacia5](https://github.com/alexenache10/agritwin-dacia5)

---

## 1. Dataset Overview & Source

- **Geographic Focus:** Căzănești agricultural area, Ialomița County, Bărăgan Plain, Romania. Flat terrain, medium-to-large arable parcels, temperate-continental climate.
- **Target Task:** Early crop classification from multi-modal satellite image time series (SITS).
- **Primary Client:** Payment and Intervention Agency for Agriculture (APIA) and crop insurance underwriters.
- **Data Sources:** Copernicus Sentinel-1 C-SAR (interferometric wide swath) and Sentinel-2 MSI (Level-2A bottom-of-atmosphere reflectance). Ground-truth annotations derived from official LPIS farmer declarations.

---

## 2. Spatial & Temporal Resolution

- **Spatial Resolution:** 10 m ground sampling distance (GSD). Patches are uniformly cut to 32 × 32 pixels, corresponding to an area of 10.24 ha per parcel footprint.
- **Temporal Span:** 5 complete annual crop seasons (`2020`, `2021`, `2022`, `2023`, `2024`), primarily covering the vegetative window from February to October.
- **Observation Frequency:** Irregular cadence due to cloud cover and orbital passes. Sequence length ranges from 3 to 31 acquisitions per parcel (mean: 14.6 in train, 14.2 in test).

---

## 3. Modalities, Channels & Units

- **Sentinel-1 (SAR GRD):** 2 polarimetric channels (VV, VH).
  - **Unit:** Radiometrically calibrated radar backscatter σ⁰ in decibels (dB), float64 format.
  - **Observed Value Range:** [-30.01 dB, +1.82 dB].
- **Sentinel-2 (MSI L2A):** 12 spectral bands (B1–B12, excluding duplicate B8A).
  - **Unit:** Surface reflectance BOA × 10⁴, uint16 format.
  - **Observed Value Range:** [470, 3370].
- **Fused Input Dimension:** 14 channels per temporal step (2 SAR + 12 MSI).

---

## 4. Volume & Class Distribution

- **Observation Units:**
  - **Train Set (`2020–2023`):** 5,431 paired S1-S2 patch observations, grouped into 373 unique parcel time series.
  - **Test Set (`2024`):** 1,023 paired S1-S2 patch observations, grouped into 72 unique parcel time series.

- **Distribution Across Crop Types (Field Series Count):**

| Crop Code | Crop Type | Train | Test |
|---|---|---:|---:|
| `101` | Winter wheat | 61 | 12 |
| `108` | Corn | 92 | 6 |
| `131` | Sunflower | 76 | 0 |
| `151` | Winter rape | 23 | 6 |
| `202` | Lucerne | 16 | 21 |
| `253` | Fodder crops | 2 | 4 |
| `254` | Dry pulses | 1 | 0 |
| `1010` | Winter barley | 27 | 4 |
| `2037` | Permanent grassland | 14 | 0 |
| `3017` | Arable land / Other crops | 46 | 16 |
| `9747` / `9748` | Fallow land | 15 | 3 |

*Note: Sunflower is absent from the 2024 test set, reportedly due to crop rotation.*

---

## 5. Data Gaps, Asynchrony & Limitations

- **S1-S2 Acquisition Delay:** Optical and radar passes are asynchronous. In the training split:
  - 0-day offset (same day): 1,434 scenes (26.4%).
  - 1-day offset: 2,269 scenes (41.8%).
  - 2-day offset: 1,338 scenes (24.6%).
  - 3-day offset: 390 scenes (7.2%).
- **Spring Cloud Gaps:** Sentinel-2 observations exhibit significant gaps during March–May. In May, train data contains only 162 valid S2 scenes versus 1,500 in August. Sentinel-1 backscatter provides uninterrupted physical structure tracking across cloud events.
- **Nomenclature Shift:** Fallow land transitions from code `9747` (2020–2021) to `9748` (2022–2024).

---

## 6. Partitioning Protocol (Splits)

- **Out-of-Year Temporal Split:** Train on `2020–2023` (373 parcels), evaluate on `2024` (72 parcels). Mimics real-world operational inference on ongoing agricultural campaigns.
- **Internal Validation:** Leave-One-Year-Out (LOYO) cross-validation across the 2020–2023 cohorts, grouped strictly by parcel UID to eliminate spatial-temporal leakage.

---

## 7. Licensing & Attribution

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Access:** Freely accessible via Big Earth Data / Taylor & Francis repository.
