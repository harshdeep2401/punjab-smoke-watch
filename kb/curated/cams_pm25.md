---
id: cams_pm25
title: What the CAMS PM2.5 values are (modelled, not measured)
doc_type: curated
as_of: 2022-01-01
source_url: https://open-meteo.com/en/docs/air-quality-api
---

# Where our PM2.5 comes from

The PM2.5 values in this project come from the Open-Meteo Air Quality API, which serves data from the Copernicus Atmosphere Monitoring Service (CAMS), run by ECMWF. We request hourly values at the coordinates of four cities (Ludhiana, Amritsar, Patiala, Bathinda) and average them to a daily mean in µg/m³.

# Modelled, not measured

CAMS global data are the output of an atmospheric model that is combined with satellite observations. They are not readings from a monitor on the street. The model grid is coarse, with cells that cover many kilometres. One value therefore stands for a wide area and can differ from a ground monitor in the same city, especially on days with local pollution peaks. Treat the values as a regional signal. They are not a health measurement for one place.

# Fire emissions are already inside the model

The CAMS forecasting system takes biomass-burning emissions from the Global Fire Assimilation System (GFAS). GFAS is built from satellite fire radiative power. This means the model PM2.5 for a day already reflects fires that the model knew about. This matters for our forecast. Today's PM2.5 may already carry much of the fire signal, so adding our own fire features can give little extra information. The model card reports what we actually measured.

# Coverage

For Ludhiana the hourly series is complete for the Sep 15 to Nov 30 windows of 2022 to 2025. Earlier years return no values in the API, which limits how many seasons we can train on.
