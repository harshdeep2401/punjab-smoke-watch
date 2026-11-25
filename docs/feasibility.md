# Data feasibility (Phase 0)

Checked on 2026-11-25 with `scripts/feasibility.py`. All figures below are measured, except where marked **PENDING**.

| Source | Usable range | Caveats |
|---|---|---|
| Open-Meteo Air Quality (CAMS), `pm2_5`, Ludhiana | Earliest non-null ≈ **2022-08-02**. 2020 and 2021 return 0 non-null hours. Seasons 2022 to 2025 are 1848/1848 hourly values (Sep 15 to Nov 30). 2026 is 1704/1704 up to 2026-11-24. | Modelled reanalysis/forecast (CAMS global, coarse grid), not ground monitors. It is not CPCB data. |
| Open-Meteo Archive weather (`temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `boundary_layer_height`) | All 5 variables fully non-null for every year tested (2020 to 2026). Latest available day was **yesterday** (lag 1 day, not the ~5 days assumed in the plan). | The 1-day lag was measured on 2026-11-25 and could vary. Phase 1 should still fall back to the forecast API with `past_days` if the archive returns nulls. |
| NASA FIRMS VIIRS S-NPP | **`VIIRS_SNPP_SP` (archive):** returns data for 1 Nov of 2021 to 2025 (21105, 13986, 8298, 2250, 2889 rows for a 5-day window) and for 2026-04 and 2026-01, so it holds the archive up to some point in 2026. It returned **0 rows for 2026-11-01**, so it does not cover the 2026 season. **`VIIRS_SNPP_NRT`:** 0 rows for every 1 Nov before 2026. For 2026 it has data from at least 2026-09-15 (earliest not found; 2026-08-01 returned 0) up to **about 2026-11-02**, and then 0 rows for every window tried from 2026-11-04 to 2026-11-20. | **Max `DAY_RANGE` is 5** (10 gives HTTP 400 `Expects [1..5]`). So the ingest must make one call per 5-day chunk. The NRT feed looks stale after about 2026-11-02, so 2026 fire data is incomplete. Each row has `acq_date` and `acq_time` (UTC) as stated in AGENTS.md. |
| District boundaries | 22 Punjab districts, geoBoundaries IND ADM2 (gbOpen, 2021 release). | **No Malerkotla** (the 23rd district, created in 2021). See below. |

## District source (decision D10)

Both candidate sources have 22 districts and neither has Malerkotla:

- (a) `udit-001/india-maps-data` Punjab file: 22 districts, census 2011 boundaries plus Fazilka/Pathankot/Tarn Taran/Barnala. The repo has no licence file (404), so reuse terms are unclear.
- (b) **geoBoundaries IND ADM2 (chosen):** 22 Punjab districts. Licence: **Open Data Commons Open Database License (ODbL) 1.0**, attribution to geoBoundaries (wmgeolab). Source: `https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM2/geoBoundaries-IND-ADM2.geojson`.
- (c) DataMeet 2011 URL returned 404, and it would have 20 or so older districts anyway.

Saved to `data/geo/punjab_districts.geojson` (filtered to Punjab, simplified with tolerance 0.005°, one property `district`). Names were normalised: `Firozpur` → `Ferozepur`, `Muktsar` → `Sri Muktsar Sahib`, `Sahibzada Ajit Singh Nagar` → `SAS Nagar (Mohali)`, `Shahid Bhagat Singh Nagar` → `SBS Nagar (Nawanshahr)`.

**Limitation to state in the app:** Malerkotla (carved out of Sangrur in 2021) is part of Sangrur here.

## Seasons

Air quality allows 2022 to 2026. FIRMS SP covers 2022 to 2025 in full. 2026 has fires only up to about Nov 2, which is about 48 of 77 season days, so it is not "most of it".

**`SEASONS = [2022, 2023, 2024, 2025]`.** That is 4 seasons, so the rule applies: test on 2024 and 2025 (expanding window). The 2024 fold trains on 2022 to 2023, and the 2025 fold trains on 2022 to 2024. The training sets are small, so say so in the model card.

2026 is excluded from training and the backtest. If the 2026 NRT feed resumes, it can be used as a partial live-style season (stretch). Ingest should use `VIIRS_SNPP_SP` for 2022 to 2025 (D3) and `VIIRS_SNPP_NRT` only for 2026.
