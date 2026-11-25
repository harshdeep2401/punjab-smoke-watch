---
id: how_to_read_forecast
title: How to read the p10 / p50 / p90 forecast
doc_type: curated
as_of: 2022-01-01
source_url: docs/PLAN.md#phase-3
---

# What the three numbers are

For each city and each day we forecast the mean PM2.5 of the next day, in µg/m³. We give three numbers, which are quantiles of the forecast:

- **p50** is the median, our central guess. Half of the outcomes should fall below it.
- **p10** is the low value. We expect the true value to fall below it about 10% of the time.
- **p90** is the high value. We expect the true value to fall above it about 10% of the time.

# The 80% interval

The range from p10 to p90 is an 80% prediction interval. If the forecast is well calibrated, the real next-day value falls inside it on about 80 of every 100 days. Coverage is the share of days that actually did. The model card reports the coverage we measured on the test seasons. A wide interval means the model is unsure. A narrow one means it is more sure, but only if the coverage is still close to 80%.

# How the interval is calibrated

The raw quantile model tends to give intervals that are too narrow on new data. We widen the interval with a simple correction measured on the last training season (conformalised quantile regression). The width therefore reflects how wrong the model was on data it had not seen.

# Reading it correctly

- The p50 is not a promise. Use the interval, not only the middle number.
- The forecast is for the model PM2.5 of a city, not for a street or a monitor.
- A forecast has no use as a health or regulatory instruction.
- Always compare with the persistence baseline (tomorrow equals today) in the model card to see whether the model adds anything.
