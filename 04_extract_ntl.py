"""
STEP 4 - Night-time-light features from the VIIRS tifs you already downloaded.
Same definitions as the paper:
    GTLI = sum of pixel radiance inside the cell   (Grid Total Light Index)
    GALI = GTLI / number of pixels                  (Grid Average Light Index)
We also keep max and standard deviation (used in the extended model).
Only pixels INSIDE West Bengal are used (cells are clipped to the border).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from config import *

grid = gpd.read_file(GRID_GPKG, layer="grid")[["Grid_ID", "geometry"]]
out = pd.DataFrame({"Grid_ID": grid["Grid_ID"]})


def pixels(src, geom):
    for touched in (False, True):          # fall back to all_touched for tiny cells
        try:
            data, _ = rasterio.mask.mask(src, [geom], crop=True,
                                         filled=False, all_touched=touched)
        except ValueError:                 # cell does not overlap the raster
            return np.array([])
        arr = data[0]
        v = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if src.nodata is not None:
            v = v[v != src.nodata]
        if len(v):
            return np.clip(v, 0, None)     # radiance cannot be negative
    return np.array([])


for year, tif in NTL_TIFS.items():
    if not tif.exists():
        raise FileNotFoundError(f"{tif} not found. Copy the VIIRS tif into data_raw/.")
    with rasterio.open(tif) as src:
        g = grid.to_crs(src.crs)
        tot, avg, mx, sd = [], [], [], []
        for geom in g.geometry:
            v = pixels(src, geom)
            if len(v):
                tot.append(v.sum()); avg.append(v.mean())
                mx.append(v.max()); sd.append(v.std())
            else:
                tot.append(np.nan); avg.append(np.nan)
                mx.append(np.nan); sd.append(np.nan)
    out[f"gtli_{year}"] = tot
    out[f"gali_{year}"] = avg
    out[f"ntl_max_{year}"] = mx
    out[f"ntl_std_{year}"] = sd
    print(f"NTL extracted for {year}")

out.to_csv(NTL_CSV, index=False)
print("\nCells with missing NTL:", int(out["gali_2021"].isna().sum()))
print("Saved:", NTL_CSV)
