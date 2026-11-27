---
id: firms_detection
title: How satellite fire detection (VIIRS, NASA FIRMS) works and what it misses
doc_type: curated
as_of: 2022-01-01
source_url: https://www.earthdata.nasa.gov/data/tools/firms/faq
---

# How satellite fire detection works

NASA FIRMS distributes active fire detections from the VIIRS instrument on the Suomi NPP (S-NPP) satellite. This project uses only S-NPP, so every year is seen by the same sensor. The VIIRS active fire product has a pixel size of about 375 m. Each detection is a pixel that looks hotter than its surroundings in the thermal infrared bands. Each record has a date, a time (UTC), a position, a fire radiative power (FRP, in megawatts) and a confidence class of low, nominal or high. We drop the low-confidence records.

# What a detection is, and is not

A detection is a heat anomaly seen at one moment. It does not say what was burning. In Punjab in October and November most such anomalies are crop residue fires, but the satellite cannot tell this from other burning. FRP is the radiative power at the instant of the overpass. It is not the total energy of the fire and not the area burned.

# Why fires are missed

- **Overpass timing.** S-NPP has nominal equator-crossing times of 13:30 and 01:30 mean local time. Because neighbouring orbits overlap, a place in the mid-latitudes may be seen 3 to 4 times a day, but always close to those two times. A fire that starts and burns out between passes is never seen.
- **Cloud and thick smoke.** The sensor cannot see through thick cloud, so fires under it are missed.
- **Small or weak fires.** Small, short or low-intensity burns can fall below the detection threshold of a 375 m pixel.

So a detection count is a lower bound that moves with the weather and the overpass timing. It is not a census of fires.

# How this project treats the dates

FIRMS times are in UTC. We use the `acq_date` as the local day. This is a simplification: the night-time pass at about 01:30 local time falls on the previous UTC date, so those detections are filed one day early. We state this as a limitation.
