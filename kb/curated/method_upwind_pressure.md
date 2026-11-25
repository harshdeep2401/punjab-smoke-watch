---
id: method_upwind_pressure
title: Method - the upwind fire-pressure feature
doc_type: curated
as_of: 2022-01-01
source_url: AGENTS.md#7-decisions-log-append-dont-rewrite
---

# The idea

Fires only matter for a city if the wind carries their smoke toward it. A large fire to the west of Ludhiana matters on a day with a west wind and matters little on a day with an east wind. The upwind fire-pressure feature turns this into one number per city per day.

# The formula in words

For each city and day, look at every fire detection within 300 km of the city. Give each fire a weight and add up FRP times weight over all fires. The weight is the product of two parts:

1. **A direction part.** Take the direction the wind is blowing toward, and the direction of the line from the fire to the city. If the two point the same way, the part is 1. As the angle between them grows, the part falls like the cosine of the angle. It is set to 0 when the wind blows away from the city (angles beyond 90 degrees).
2. **A distance part.** exp(-d / 150), where d is the distance in km. A fire 150 km away counts about 37% as much as one right next to the city.

Fire radiative power (FRP) is in megawatts, so a bigger or hotter fire counts more.

We compute this for the same day (lag 0) and for the two previous days (lags 1 and 2), because smoke takes time to travel and to build up.

# Wind input

The daily wind direction is the vector mean of the daily mean east-west and north-south wind components, not a plain average of the angles. This avoids wrong results around the 0/360 degree wrap. Wind direction follows the meteorological convention: the direction the wind blows from.

# What the feature leaves out

It is an explainable score, not an atmospheric dispersion model. It uses one daily mean wind per city, so it ignores wind shifts within a day and at different heights. It ignores the boundary layer height, rain, and fires beyond 300 km. It also inherits all the detection gaps of satellite fire data (see the satellite detection note).
