---
id: mcd64a1
title: The MODIS burned-area product (MCD64A1) and what it misses
doc_type: curated
part: 2
as_of: 2022-01-01
source_url: https://lpdaac.usgs.gov/documents/1006/MCD64_User_Guide_V61.pdf
---

# What the product is

MCD64A1 is NASA's monthly burned-area map on a grid of about 500 m. According to its Collection 6.1 user guide, the algorithm uses 500 m MODIS images together with 1 km MODIS active-fire observations. Maps of active fires guide the choice of training samples and the prior probabilities. It records the day each grid cell burned.

So it is less tied to the moment of a satellite pass than a fire count, because a burn scar can still be seen after the flames are out. It is not fully independent of overpass times, because it leans on MODIS active fires.

# How uncertain it is

The user guide's global validation (section 7) estimated a 72.6% omission error and a 40.2% commission error, and says the product systematically underestimates small burned areas. Section 8.1 says burned areas in cropland should generally be treated as low confidence, because farm burning is hard to map reliably.

These are global and general statements. This project has not measured the product's error in Punjab. Punjab's paddy fields are exactly the kind of burning the guide warns about, so this error is the largest unknown in any result that uses burned area.
