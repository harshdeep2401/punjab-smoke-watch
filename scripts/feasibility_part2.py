"""Phase 8 feasibility checks (Part 2: Audit + Response). Prints measurements; never prints API keys.

    uv run python scripts/feasibility_part2.py firms rain
    uv run --with pystac-client --with planetary-computer --with rasterio \
        python scripts/feasibility_part2.py burned s2
    uv run --with rasterio --with paramiko python scripts/feasibility_part2.py burned_umd

`burned`, `burned_umd` and `s2` need packages that are not project dependencies yet (AGENTS.md §3 adds them
only if this check says go), hence the `--with` flags.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

from psw import config
from psw.ingest import firms, http
from psw.processing.districts import assign_districts

GEO = config.GEO / "punjab_districts.geojson"
YEARS = [2022, 2023, 2024, 2025]
SENSORS = ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "MODIS"]
AVAIL_URL = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{key}/all"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"
MODIS_PIXEL_KM2 = 0.463312717**2  # 500 m sinusoidal grid (equal area)


# ---------------------------------------------------------------- FIRMS sensors


def ist_hours(acq_time: pd.Series) -> pd.Series:
    """FIRMS acq_time is HHMM in UTC; IST = UTC + 5:30. Returns the IST hour of day (0-23)."""
    t = acq_time.astype(int)
    return (((t // 100) * 60 + t % 100 + 330) % 1440) // 60


def hour_summary(df: pd.DataFrame) -> str:
    c = Counter(ist_hours(df["acq_time"]))
    return " ".join(f"{h:02d}h:{c[h]}" for h in sorted(c))


def pass_times(df: pd.DataFrame) -> str:
    """Median and 5th-95th percentile IST clock time of the day (08-18 h) and night detections."""
    t = df["acq_time"].astype(int)
    minutes = ((t // 100) * 60 + t % 100 + 330) % 1440
    day = (minutes >= 8 * 60) & (minutes < 18 * 60)
    parts = []
    for name, m in (("day", minutes[day]), ("night", minutes[~day])):
        if len(m):
            p05, p50, p95 = (f"{int(v) // 60:02d}:{int(v) % 60:02d}" for v in m.quantile([0.05, 0.5, 0.95]))
            parts.append(f"{name} n={len(m)} median {p50} (p05 {p05}, p95 {p95})")
    return "; ".join(parts)


def check_firms() -> None:
    print("== FIRMS data availability (min and max date per product) ==")
    key = config.FIRMS_MAP_KEY
    r = http.get(AVAIL_URL.format(key=key), secret=key)
    print(r.text.replace(key, "***"))

    print("== FIRMS 5-day windows from 1 Nov, bbox (all) and inside Punjab ==")
    for sensor in SENSORS:
        for year in YEARS:
            for suffix in ("SP", "NRT"):
                source = f"{sensor}_{suffix}"
                try:
                    df = firms.fetch_chunk(source, date(year, 11, 1), 5, key)
                except RuntimeError as e:
                    print(f"{source} {year}: ERROR {e}")
                    continue
                if df.empty:
                    print(f"{source} {year}: 0 rows")
                    continue
                df = df.reset_index(drop=True)
                inside = assign_districts(df, GEO)
                inside = inside[inside["in_punjab"]]
                print(f"{source} {year}: {len(df)} rows, {len(inside)} in Punjab")
                for sat, g in df.groupby(df["satellite"].astype(str)):
                    pg = inside[inside["satellite"].astype(str) == sat]
                    print(f"    satellite={sat} (bbox n={len(g)}) IST hours: {hour_summary(g)}")
                    print(f"        pass times: {pass_times(g)}")
                    if len(pg):
                        print(f"        Punjab only (n={len(pg)}): {hour_summary(pg)}")
                if "MODIS" in sensor:
                    conf = pd.to_numeric(df["confidence"])
                    print(f"    MODIS confidence: min {conf.min()}, median {conf.median()}, max {conf.max()}, "
                          f"share < 30: {(conf < 30).mean():.1%}")
                break  # SP had data: no need to try NRT

    print("== FIRMS burned-area products through the area API (probe) ==")
    bbox = ",".join(str(x) for x in config.BBOX)
    for source in ("BA_MODIS", "BA_VIIRS"):
        url = firms.URL.format(key=key, source=source, bbox=bbox, day_range=5, start="2024-11-01")
        r = http.get(url, secret=key)
        print(source, "HTTP", r.status_code, "data rows:", len(r.text.strip().split("\n")) - 1,
              "header:", repr(r.text.split("\n")[0][:200].replace(key, "***")))


# ---------------------------------------------------------------- Burned area (Planetary Computer)


def _catalog():
    import planetary_computer
    from pystac_client import Client

    return Client.open(STAC, modifier=planetary_computer.sign_inplace)


def check_burned() -> None:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds

    print("== MCD64A1 (modis-64A1-061) on Planetary Computer, no key ==")
    districts = gpd.read_file(GEO).to_crs("EPSG:4326")
    cat = _catalog()
    print("licence field:", cat.get_collection("modis-64A1-061").license)
    t0 = time.time()
    for year in YEARS:
        search = cat.search(collections=["modis-64A1-061"], bbox=list(districts.total_bounds),
                            datetime=f"{year}-09-01/{year}-11-30")
        items = sorted(search.items(), key=lambda i: i.id)
        # MCD64A1.AYYYYDDD.hXXvYY.061.<processing>: keep months Sep-Nov of this year (start day 244..335)
        items = [i for i in items if i.id.split(".")[1].startswith(f"A{year}") and
                 240 <= int(i.id.split(".")[1][5:]) <= 310]
        lo = date(year, *config.SEASON_START).timetuple().tm_yday
        hi = date(year, *config.SEASON_END).timetuple().tm_yday
        totals: Counter = Counter()
        for item in items:
            t = time.time()
            with rasterio.open(item.assets["Burn_Date"].href) as src:
                shapes = districts.to_crs(src.crs)
                win = from_bounds(*shapes.total_bounds, transform=src.transform).round_offsets().round_lengths()
                try:
                    win = win.intersection(Window(0, 0, src.width, src.height))
                except rasterio.errors.WindowError:
                    print(f"  {item.id}: no overlap with Punjab")
                    continue
                arr = src.read(1, window=win)
                mask = rasterize([(g, 1) for g in shapes.geometry], out_shape=arr.shape,
                                 transform=src.window_transform(win), fill=0, dtype="uint8")
            tile, doy = item.id.split(".")[2], item.id.split(".")[1][5:]
            burned = int(((arr >= lo) & (arr <= hi) & (mask == 1)).sum())
            unmapped = int(((arr < 0) & (mask == 1)).sum())
            totals[doy] += burned
            print(f"  {item.id[:30]}: tile {tile}, Punjab pixels {int(mask.sum())}, burned {burned}, "
                  f"negative (unmapped/water) {unmapped}, {time.time() - t:.1f} s")
        per_month = {d: round(n * MODIS_PIXEL_KM2, 1) for d, n in sorted(totals.items())}
        print(f"{year}: {len(items)} items; Punjab burned km2 (Sep 15-Nov 30) by file start day {per_month}, "
              f"total {sum(totals.values()) * MODIS_PIXEL_KM2:.1f} km2")
    print(f"all MCD64A1 reads: {time.time() - t0:.1f} s")


def check_s2(district: str = "Sangrur", year: int = 2024, res: int = 80) -> None:
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    print(f"== Sentinel-2 dNBR, {district} {year}, {res} m ==")
    t0 = time.time()
    shp = gpd.read_file(GEO)
    shp = shp[shp["district"] == district].to_crs("EPSG:32643")
    xmin, ymin, xmax, ymax = shp.total_bounds
    width, height = int((xmax - xmin) // res) + 1, int((ymax - ymin) // res) + 1
    transform = from_origin(xmin, ymax, res, res)
    mask = rasterize([(g, 1) for g in shp.geometry], out_shape=(height, width), transform=transform,
                     fill=0, dtype="uint8").astype(bool)
    cat = _catalog()
    bbox = list(shp.to_crs("EPSG:4326").total_bounds)

    def read(href: str, resampling) -> np.ndarray:
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs="EPSG:32643", transform=transform, width=width, height=height,
                           resampling=resampling, src_nodata=0, nodata=0) as vrt:
                return vrt.read(1)

    def composite(start: str, end: str) -> np.ndarray:
        items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                                query={"eo:cloud_cover": {"lt": 30}}).items())
        items.sort(key=lambda i: i.properties["eo:cloud_cover"])
        out = np.full((height, width), np.nan, dtype="float32")
        used = 0
        for item in items:
            need = mask & np.isnan(out)
            if need.sum() < 0.01 * mask.sum():
                break
            scl = read(item.assets["SCL"].href, Resampling.nearest)
            # keep vegetation (4), bare soil (5), water (6), unclassified (7)
            ok = need & np.isin(scl, [4, 5, 6, 7])
            if not ok.any():
                continue
            offset = 1000 if float(item.properties.get("s2:processing_baseline", "0")) >= 4 else 0
            nir = read(item.assets["B8A"].href, Resampling.average).astype("float32") - offset
            swir = read(item.assets["B12"].href, Resampling.average).astype("float32") - offset
            ok &= (nir + swir) > 0
            out[ok] = (nir[ok] - swir[ok]) / (nir[ok] + swir[ok])
            used += 1
        valid = (~np.isnan(out) & mask).sum() / mask.sum()
        print(f"  {start}..{end}: {len(items)} items under 30% cloud, {used} used, "
              f"valid cover {valid:.1%}, {time.time() - t0:.0f} s elapsed")
        return out

    with rasterio.Env(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        pre = composite(f"{year}-10-01", f"{year}-10-10")
        post = composite(f"{year}-11-20", f"{year}-11-30")
    d = (pre - post)[mask]
    d = d[~np.isnan(d)]
    area = mask.sum() * res * res / 1e6
    print(f"  district area {area:.0f} km2; valid dNBR pixels {d.size / mask.sum():.1%}")
    for thr in (0.1, 0.27, 0.44):
        print(f"  share of valid pixels with dNBR > {thr}: {(d > thr).mean():.1%}")
    print(f"  median NBR early Oct {np.nanmedian(pre[mask]):.2f}, late Nov {np.nanmedian(post[mask]):.2f}")
    print(f"  total time {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- Burned area (University of Maryland SFTP)

# Server and shared login are published in the MCD64A1 Collection 6.1 User's Guide, section 4.1
# (https://lpdaac.usgs.gov/documents/1006/MCD64_User_Guide_V61.pdf): "we openly share the password".
UMD_HOST, UMD_USER, UMD_PASSWORD = "fuoco.geog.umd.edu", "fire", "burnt"
UMD_DIR = "data/MODIS/C61/MCD64A1/TIFF/Win18/{year}"  # window 18 = South Asia (60-93 E, 5-36 N)


def check_burned_umd() -> None:
    import geopandas as gpd
    import paramiko
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds

    print("== MCD64A1 GeoTIFF window 18 from the University of Maryland SFTP server ==")
    districts = gpd.read_file(GEO).to_crs("EPSG:4326")
    cache = config.RAW / "burned" / "mcd64_win18"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    transport = paramiko.Transport((UMD_HOST, 22))
    transport.connect(username=UMD_USER, password=UMD_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        for year in YEARS:
            lo = date(year, *config.SEASON_START).timetuple().tm_yday
            hi = date(year, *config.SEASON_END).timetuple().tm_yday
            remote = UMD_DIR.format(year=year)
            names = sorted(n for n in sftp.listdir(remote) if n.endswith("burndate.tif"))
            months = {date(year, m, 1).timetuple().tm_yday for m in (9, 10, 11)}
            wanted = [n for n in names if int(n.split(".")[1][5:]) in months]
            per_month = {}
            for name in wanted:
                local = cache / name
                if not local.exists():
                    sftp.get(f"{remote}/{name}", str(local) + ".part")
                    (cache / f"{name}.part").rename(local)
                with rasterio.open(local) as src:
                    win = from_bounds(*districts.total_bounds, transform=src.transform)
                    win = win.round_offsets().round_lengths()
                    arr = src.read(1, window=win)
                    tr = src.window_transform(win)
                mask = rasterize([(g, 1) for g in districts.geometry], out_shape=arr.shape, transform=tr,
                                 fill=0, dtype="uint8").astype(bool)
                # lat/lon grid: pixel area shrinks with latitude
                lat = tr.f + (np.arange(arr.shape[0]) + 0.5) * tr.e
                px_km2 = (abs(tr.e) * 111.195) * (tr.a * 111.195 * np.cos(np.radians(lat)))
                area = np.broadcast_to(px_km2[:, None], arr.shape)
                burned = area[mask & (arr >= lo) & (arr <= hi)].sum()
                per_month[name.split(".")[1]] = round(float(burned), 1)
                print(f"  {name}: {local.stat().st_size / 1e6:.2f} MB, Punjab area {area[mask].sum():.0f} km2, "
                      f"burned {burned:.1f} km2")
            print(f"{year}: {len(wanted)}/3 months {per_month}, total {sum(per_month.values()):.1f} km2")
    finally:
        transport.close()
    print(f"UMD total: {time.time() - t0:.1f} s")


# ---------------------------------------------------------------- Rain at district centroids


def district_centroids() -> dict[str, tuple[float, float]]:
    import geopandas as gpd

    g = gpd.read_file(GEO)
    pts = g.to_crs("EPSG:32643").centroid.to_crs("EPSG:4326")
    return {d: (round(p.y, 3), round(p.x, 3)) for d, p in zip(g["district"], pts)}


def check_rain() -> None:
    print("== Open-Meteo archive precipitation_sum at district centroids ==")
    cents = district_centroids()
    calls, t0 = 0, time.time()
    for year in YEARS:
        nonnull = total = 0
        rain = {}
        for district, (lat, lon) in cents.items():
            r = http.get(WX_URL, {"latitude": lat, "longitude": lon, "daily": "precipitation_sum",
                                  "start_date": f"{year}-09-15", "end_date": f"{year}-11-30",
                                  "timezone": "Asia/Kolkata"})
            calls += 1
            if r.status_code != 200:
                print(district, year, "HTTP", r.status_code, r.text[:100])
                continue
            v = r.json()["daily"]["precipitation_sum"]
            total += len(v)
            nonnull += sum(x is not None for x in v)
            rain[district] = sum(x or 0 for x in v)
            time.sleep(0.2)
        vals = np.array(list(rain.values()))
        print(f"{year}: {len(rain)} districts, {nonnull}/{total} non-null district-days; season rain mm "
              f"min {vals.min():.0f}, median {np.median(vals):.0f}, max {vals.max():.0f}")
    print(f"{calls} calls in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    which = sys.argv[1:] or ["firms", "rain"]
    for w in which:
        {"firms": check_firms, "burned": check_burned, "burned_umd": check_burned_umd, "s2": check_s2,
         "rain": check_rain}[w]()
