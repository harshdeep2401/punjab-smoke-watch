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

---

# Part 2: data for Audit + Response (Phase 8)

Checked on 2026-11-26. Satellite, burned-area and rain figures are measured by `scripts/feasibility_part2.py`. The source checks for official counts, mandi arrivals and machinery were done by hand (pages opened, addresses below).

## Go / no-go

| Signal | Source used | Seasons | Access | Verdict | Main caveat |
|---|---|---|---|---|---|
| VIIRS NOAA-20 | FIRMS `VIIRS_NOAA20_SP` | 2022 to 2025 | FIRMS key | **GO** | Passes at almost the same time as S-NPP |
| VIIRS NOAA-21 | FIRMS `VIIRS_NOAA21_SP` | **2023 to 2025** | FIRMS key | **GO** | No 2022 (SP starts 2023-03-01), so its base season is 2023 |
| MODIS Aqua | FIRMS `MODIS_SP` | 2022 to 2025 | FIRMS key | **GO** | Pass time drifts from about 14:00 to about 15:15 IST |
| MODIS Terra | FIRMS `MODIS_SP` | 2022 to 2025 | FIRMS key | **GO, weak** | Very few detections (4 to 102 in Punjab per 5-day window); drifts earlier |
| Burned area, MCD64A1 | University of Maryland SFTP, GeoTIFF window 18 | 2022 to 2025, Sep to Nov | Shared login published in the user guide | **GO** (not the planned route) | Low confidence in cropland; uses active fires |
| Burned area, MCD64A1 | Planetary Computer STAC `modis-64A1-061` (D17 default) | Months missing | No key | **NO-GO** | No Sep 2023, no Sep or Oct 2025 |
| Burned area, Sentinel-2 dNBR | Planetary Computer `sentinel-2-l2a` | Tested 2024 | No key | **NO-GO** | Measures the paddy harvest, not burning |
| Official counts, district | PPCB reports, some filed at the National Green Tribunal | 2022 to 2025 | Public PDF | **GO** | Satellite-based; hand-entered; two source documents |
| Mandi paddy arrivals | Agmarknet, data.gov.in | none | Refused (HTTP 403) | **NO-GO** | Use the Phase 11 proxies |
| CRM machines by district | Punjab status reports at the National Green Tribunal | 2018-19 to 2024-25 | Public PDF | **GO** (found 2026-11-27) | Machines supplied on subsidy, not machines in working order |
| District rain | Open-Meteo archive `precipitation_sum` | 2022 to 2025 | No key | **GO** | Reanalysis, not rain gauges |

**Chosen burned-area path:** MCD64A1 monthly `burndate` GeoTIFFs from the University of Maryland server. This changes D17 (which named Planetary Computer), so Phase 9 needs `rasterio` and `paramiko` instead of `pystac-client`, `planetary-computer` and `rasterio`. **Approved by the user on 2026-11-26 (D22) and built in Phase 9.**

## 1. FIRMS sensors

FIRMS data-availability endpoint: `MODIS_SP`, `VIIRS_NOAA20_SP`, `VIIRS_NOAA21_SP` and `VIIRS_SNPP_SP` all run to 2026-08-31, so SP covers every past season and NRT is not needed for 2022 to 2025.

Detections inside Punjab, 5 days from 1 November (one window per year, not season totals):

| Year | S-NPP | NOAA-20 | NOAA-21 | MODIS (of which Terra) |
|---|---|---|---|---|
| 2022 | 12,343 | 13,636 | no data | 2,738 (102) |
| 2023 | 7,127 | 8,337 | 8,866 | 2,211 (17) |
| 2024 | 1,200 | 2,053 | 1,843 | 681 (4) |
| 2025 | 922 | 989 | 1,010 | 796 (6) |

Median pass time in IST over the whole bounding box (day detections, 08:00 to 18:00):

| Satellite | 2022 | 2023 | 2024 | 2025 | Night median |
|---|---|---|---|---|---|
| S-NPP | 13:45 | 13:45 | 13:56 | 13:43 | about 02:00 to 02:30 |
| NOAA-20 | 13:38 | 13:32 | 13:43 | 13:38 | about 02:00 to 02:30 |
| NOAA-21 | no data | 13:37 | 13:33 | 13:32 | about 02:00 to 02:15 |
| Aqua | 13:55 | 13:59 | 14:34 | **15:15** | few detections |
| Terra | 11:10 | 10:51 | 10:34 | 10:24 | about 21:00 to 22:00, very few |

Findings that change the plan's assumptions:
- **The three VIIRS satellites do not add different times of day.** Their median day passes are within about 25 minutes of each other in every year (usually under 15), not 50 minutes apart in clock time. They add repeat looks near 13:30, so they share the same blind spot.
- **Only Aqua (later each year) and the night passes see other hours.** Terra's drift to earlier and Aqua's drift to later change what each can catch, so a change in MODIS counts is partly a change of observing time. NASA describes [Terra's drift](https://terra.nasa.gov/about/terra-orbital-drift-information). The Aqua drift here is measured; the NASA-side page for it did not load.
- Phase 9 `pass_label` windows must be wide enough: morning about 10:00 to 11:45, afternoon about 12:00 to 15:45, night about 00:45 to 03:15, and Terra's evening passes fall in "other".
- MODIS confidence runs 0 to 100, median about 63. Dropping `< 30` removes 2 to 5% of rows.
- `BA_MODIS` and `BA_VIIRS` appear in the availability list, but the area API returned fire-detection columns and 0 rows for both windows tried, so they are not a burned-area route.

## 2. Burned area

**MCD64A1 from the University of Maryland** (burn dates 15 Sep to 30 Nov, inside the 22 district polygons, 12 files of about 2.3 MB, 67 s to download and read):

| Season | Sep file | Oct file | Nov file | Total km² |
|---|---|---|---|---|
| 2022 | 30.7 | 3,318.8 | 13,187.9 | 16,537.4 |
| 2023 | 192.1 | 5,822.6 | 8,983.2 | 14,997.9 |
| 2024 | 72.2 | 704.7 | 5,570.5 | 6,347.4 |
| 2025 | 68.9 | 3,860.1 | 5,173.2 | 9,102.2 |

The polygons cover 50,502 km² in this grid, close to Punjab's area.

- **Cross-check:** Planetary Computer read with no key (about 1 s per file, 20 s in all) and agrees within 0.2% where it has all the files (2022: 16,554.9; 2024: 6,343.6). But it lacks September 2023 and September and October 2025, which NASA's catalogue does list. Its 2025 total (5,174.8) would be 43% too low with no error raised, so **Phase 9 must fail loudly if a month is missing.**
- **Cropland warning from the [C6.1 user guide](https://lpdaac.usgs.gov/documents/1006/MCD64_User_Guide_V61.pdf), §8.1:** "Burned areas in cropland should generally be treated as low confidence due to the inherent difficulty in mapping agricultural burning reliably." §7 reports a global 72.6% omission error, mainly from small burns.
- **Not independent of overpass times:** §2 says the algorithm uses 500 m imagery "coupled with 1-km MODIS active fire observations", and that active-fire maps guide the training samples and prior probabilities. A burn scar can still be seen after the fire is out, but this signal leans on Terra and Aqua fire detections.
- **Server:** host, login and password are printed in §4.1 of the guide, which says "we openly share the password".

**Sentinel-2 dNBR** (Sangrur, 80 m, 2024, early October against late November): it ran in 222 s with 99.8% valid pixels, so it is feasible to compute. But dNBR > 0.27 covers **86.6%** of the district and > 0.1 covers 93.8%, with median NBR falling from 0.59 to -0.10. Green standing paddy in early October against harvested fields in late November looks like burning. A usable version needs image pairs a few days apart and field-level cloud work, which is out of scope. **No-go.**

## 3. Official counts

| Season | State total | District table | Address |
|---|---|---|---|
| 2022 | 49,922 | **all 23 districts**, sum matches | [PPCB action taken report, 19 Mar 2024](https://www.greentribunal.gov.in/sites/default/files/news_updates/Action%20Taken%20Report%20on%20behalf%20of%20Punjab%20Pollution%20Control%20Board%20in%20OA%20No.%20632%20of%202023%20%28In%20re%20%20News%20item%20appearing%20in%20Hindu%20dated%2006.10.2023%20....pdf), PDF page 57 |
| 2023 | 36,663 | **all 23 districts**, sum matches | [PPCB action taken report, 19 Mar 2024](https://www.greentribunal.gov.in/sites/default/files/news_updates/Action%20Taken%20Report%20on%20behalf%20of%20Punjab%20Pollution%20Control%20Board%20in%20OA%20No.%20632%20of%202023%20%28In%20re%20%20News%20item%20appearing%20in%20Hindu%20dated%2006.10.2023%20....pdf), PDF page 57 |
| 2024 | 10,909 | **all 23 districts**, sum matches | https://ppcb.punjab.gov.in/media/documents/ATR_30.11.2024.pdf |
| 2025 | 5,114 | **all 23 districts**, sum matches | https://ppcb.punjab.gov.in/media/documents/CAQM_ATR_30.11.2025.pdf |

- PPCB's 2022 and 2023 report index pages return 404. The 2022 and 2023 district figures come from section 7.3 of the action plan annexed to the report above (PRSC satellite data, 15 Sep to 30 Nov). The same page has 2021 (71,159). Its 2022 column matches the one in PPCB's 2023 action plan, district for district. A [journal paper](https://neptjournal.com/upload-images/(40)B-4139.pdf) has a different series (59,166 and 43,648) that must not be mixed with it.
- **How they are counted:** the 2024 table says incidents "reported by PRSC" (Punjab Remote Sensing Centre); the 2025 table says "as per ISRO protocol". Both are satellite-based, so they are **not independent of overpass times**. The reports do not say which sensors.
- The reports also give field visits to the **detected** points (2024: 10,791 visited, 5,090 with no crop burning observed, marked as under verification; 2025: 5,054 inspected, 2,391 with burning verified physically). The two columns are defined differently. Neither says anything about burning that was never detected.
- Malerkotla appears as its own row and must be added to Sangrur (D10).
- **Licence:** the PPCB pages and PDFs state none. Cite them with the address on each row. No numbers have been entered into the repo yet.

## 4. Harvest-progress signal: no-go

- `api.agmarknet.gov.in/v1/daily-price-arrival/filters` returned **HTTP 403** to a plain script client. I did not disguise the client.
- data.gov.in's Punjab arrivals catalog page returned HTTP 403, so its granularity and years are unchecked.
- A [third-party extract](https://github.com/herrrickshaw/india-mandi-arrivals) has daily market-level rows only from November 2025. Its notes say the main daily arrivals report on Agmarknet needs a captcha.
- Phase 11 therefore uses the proxies (season progress, days since first fire, last season's curve, rain). A person could still download arrivals from the Agmarknet site in a browser into `data/manual/mandi_arrivals.csv`.

## 5. Residue-management machinery: go (revised 2026-11-27)

- [Status report of 26 Nov 2024](https://www.greentribunal.gov.in/sites/default/files/news_updates/Status%20Report%20of%20State%20of%20Punjab%20in%20OA%20NO%20632%20of%202023%20In%20re%20News%20item%20appearing%20in%20Hindu%20dated%2006.10.2023%20titled%20Pollution%20takes%20a%20front%20seat%20as%20stubble%20fires%20spike%20in%20Punjab.pdf), Annexure C: cumulative machines supplied on subsidy, 2018-19 to 2024-25, by type for all 23 districts (146,540 in total; the district totals add up).
- [Status report of 2 Sep 2024](https://www.greentribunal.gov.in/sites/default/files/news_updates/Status%20Report%20by%20State%20of%20Punjab%20in%20OA%20No.%20632%20of%202023%20%28In%20re%20News%20Item%20published%20in%20The%20Hindu%20dated%2006.10.2023%20titled%20as%20Pollution%20take%20a%20front%20seat.pdf), Annexure A: the same by year, 2018-19 to 2023-24.
- These count machines **supplied**, including tractors and rotavators. They do not say how many still work or are free to move. The planner can use them to inform defaults and per-district limits, but the fleet size stays a user input. No figures have been entered into the repo.

The first check (2026-11-26) had called this a no-go; it did not look at the tribunal filings.

## 6. Rain: go

88 calls (22 district centroids × 4 seasons) in 99 s, with 6,776 of 6,776 district-days non-null. Median season rain (15 Sep to 30 Nov): 2022 122 mm, 2023 94 mm, 2024 16 mm, 2025 51 mm.

## Evening burning: pages opened, for the Phase 10 notes

- [Tribune, 10 Dec 2025](https://www.tribuneindia.com/news/haryana/farmers-shift-stubble-burning-time-to-dodge-satellite-tracking-isro-study) and [ThePrint, 3 Dec 2025](https://theprint.in/environment/are-farm-fires-going-uncounted-isro-study-suggests-farmers-have-figured-out-a-way-to-evade-satellites/2797583/): report an ISRO Space Applications Centre study in *Current Science* using the geostationary MSG sensor, with peak fire time moving from about 13:30 IST (2020) to about 17:00 (2024). ThePrint also quotes a researcher who calls the evasion claim unsubstantiated. **I have not opened the paper itself.**

## Licences and terms

| Source | Terms | Where stated |
|---|---|---|
| FIRMS and MCD64A1 (NASA data) | CC0 unless marked; citation strongly urged | [NASA data use guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance) |
| University of Maryland server | Data "intended to be freely available" | MCD64 user guide, §4.1 |
| Open-Meteo | CC BY 4.0; a link is required where the data is shown | [open-meteo.com/en/licence](https://open-meteo.com/en/licence) |
| Planetary Computer | Collection licence field says "proprietary"; terms page not readable | not used |
| Sentinel-2 | not checked | not used |
| PPCB / CAQM reports | none stated | n/a |
| Press articles | publisher copyright, no reuse terms | cite and link only |
| Agmarknet, data.gov.in | not seen (refused) | not used |
