---
id: limitations
title: Limitations of this project
doc_type: curated
as_of: 2022-01-01
source_url: Punjab_Smoke_Watch.md#11-limitations-to-state-openly
---

# What to keep in mind when reading any output

**Fires.** Satellites detect heat anomalies. They do not know that a fire is stubble burning. They miss fires that burn between overpasses, fires under cloud, and small or short burns. A district's detection count is a lower bound. Fire dates are UTC dates that we treat as local days, so some night detections are filed one day early.

**Air quality.** PM2.5 is modelled by CAMS on a coarse grid. It is not measured at street level and it is not CPCB monitor data. It already includes fire emissions as the model sees them.

**Districts.** We use 22 district boundaries. Malerkotla, created in 2021, is not a separate district in our boundary data. Its area is counted in Sangrur.

**Cities.** Forecasts exist for four cities only: Ludhiana, Amritsar, Patiala and Bathinda. District briefings use the fires of a district and the PM2.5 and wind of the cities, so a district without a city has no PM2.5 forecast of its own.

**Data size.** Only four seasons (2022 to 2025) have complete fire and PM2.5 data. The forecast is tested on the last two seasons, and each test fold is trained on only two or three seasons. Results from so few seasons are noisy. Fires also fell a lot from 2022 to 2025, so the seasons are not alike.

**Forecast skill.** See the model card for the measured results, including how the model compares with a persistence baseline and with a model without fire features.

**Use.** This is a student project. It is not a regulatory or health tool and gives no medical advice. It does not identify any individual farm or farmer.
