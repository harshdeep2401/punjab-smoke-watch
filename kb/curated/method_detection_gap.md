---
id: method_detection_gap
title: How this project estimates the detection gap, and its weaknesses
doc_type: curated
part: 2
as_of: 2022-01-01
source_url: docs/PLAN.md#phase-10-audit-analytics-gap-ranges-auditor-agent
---

# What the gap measures

The detection gap is the fall in fire counts per burned square kilometre since a base season: one minus the ratio of (count divided by burned area) now to (count divided by burned area) in the base season. In plain words it compares how far counts fell with how far burned area fell. It is a relative loss of detection since the base season. It is not the share of burning that is missed, because the base season's own miss rate is unknown.

# Roles of the signals

The signals depend on each other, so each has one role. Counts, the numerator: S-NPP detections and official counts. Reference, the denominator: MODIS burned area, the only reference available. Timing indicators, for Punjab as a whole: the night share of S-NPP detections, and the change in MODIS counts compared with S-NPP. They set the confidence and never enter the range. Robustness check: NOAA-20 and NOAA-21, which pass at about the same time as S-NPP.

# Range, thresholds and confidence

There is one estimate for each count series and base season (2022 or 2023), and the range runs from the lowest to the highest. Nothing is clipped, so a negative value means burned area fell more than counts. A pair is used only if burned area is at least 50 km² in both seasons and the base count is at least 100; otherwise the result is "insufficient evidence". Confidence is never high. Punjab is medium only if the lowest estimate is at least 0.25 and both timing indicators support a shift. Districts are low at most.

# Weaknesses

The range does not include the error of the burned-area product, which is unknown and is the largest uncertainty. A fall in detections per burned area fits three explanations: burning moved outside pass times, fires became smaller or cooler, or the burned-area product's sensitivity changed. The timing indicators speak only to the first. There is no field verification. This is triangulation, not ground truth.
