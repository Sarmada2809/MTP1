"""
STEP 1 - Build the 5 km master grid of West Bengal.
  * reads the district shapefile, keeps West Bengal, dissolves it into the state outline
  * covers the state with 5 km x 5 km cells (EVERY cell, not only cells with surveys)
  * clips each cell to the state border so statistics only use pixels inside West Bengal
  * saves the grid (GeoPackage) + a shapefile you upload to Google Earth Engine
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from config import *


def fix(g):
    try:
        from shapely import make_valid
        return make_valid(g)
    except ImportError:
        return g.buffer(0)


def polys_only(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        parts = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(parts) if parts else None
    return None


d = gpd.read_file(DISTRICT_SHP)
for c in (DISTRICT_FIELD, STATE_FIELD):
    if c not in d.columns:
        raise ValueError(f"Column {c} not in shapefile. Columns: {list(d.columns)}")
if d.crs is None:
    raise ValueError("Shapefile has no CRS.")
d = d.to_crs(CRS_GEO)

key = d[STATE_FIELD].astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
wb = d.loc[key == "westbengal", [DISTRICT_FIELD, "geometry"]].rename(
    columns={DISTRICT_FIELD: "District"})
if wb.empty:
    raise ValueError("No West Bengal rows found. Check STATE_FIELD / spelling.")
wb["District"] = wb["District"].astype(str).str.strip()

_map = DATA_RAW / "district_name_map.csv"
if _map.exists():
    _m = pd.read_csv(_map, dtype=str)
    _lut = dict(zip(_m["shapefile_code"].str.strip(), _m["final_name"].str.strip()))
    wb["District"] = wb["District"].map(lambda c: _lut.get(c, c))
    print("Applied district_name_map.csv")
else:
    print("WARNING: no district_name_map.csv - run 01b_name_districts.py first")
wb["geometry"] = wb.geometry.apply(fix)
wb = wb.to_crs(CRS_PROJ).dissolve(by="District", as_index=False)
wb.to_file(DISTRICT_GPKG, layer="districts", driver="GPKG")
print("Districts in shapefile:", len(wb))
print(sorted(wb["District"]))

# state outline (simplified by 100 m so the GEE upload stays light)
state = unary_union(list(wb.geometry)).simplify(100)

G = GRID_SIZE_M
minx, miny, maxx, maxy = state.bounds
gx0, gx1 = int(np.floor(minx / G)), int(np.floor(maxx / G))
gy0, gy1 = int(np.floor(miny / G)), int(np.floor(maxy / G))

gxs, gys, geoms = [], [], []
for gx in range(gx0, gx1 + 1):
    for gy in range(gy0, gy1 + 1):
        gxs.append(gx)
        gys.append(gy)
        geoms.append(box(gx * G, gy * G, (gx + 1) * G, (gy + 1) * G))
grid = gpd.GeoDataFrame({"gx": gxs, "gy": gys}, geometry=geoms, crs=CRS_PROJ)
grid = grid[grid.intersects(state)].copy()

grid["geometry"] = [polys_only(g.intersection(state)) for g in grid.geometry]
grid = grid[grid.geometry.notna()].copy()
grid["area_km2"] = grid.geometry.area / 1e6
grid["frac_in_state"] = grid["area_km2"] / (G * G / 1e6)
grid = grid[grid["frac_in_state"] >= MIN_FRAC_IN_STATE].copy()

grid["Grid_ID"] = grid["gx"].astype(str) + "_" + grid["gy"].astype(str)
grid["spatial_block"] = ((grid["gx"] // BLOCK_CELLS).astype(str) + "_"
                         + (grid["gy"] // BLOCK_CELLS).astype(str))
grid = grid.reset_index(drop=True)

grid.to_file(GRID_GPKG, layer="grid", driver="GPKG")

up = DATA_PROC / "gee_upload"
up.mkdir(exist_ok=True)
grid[["Grid_ID", "geometry"]].to_crs(CRS_GEO).to_file(up / "WB_grid_5km.shp")

print("\nTotal 5 km cells covering West Bengal:", len(grid))
print("Spatial blocks (20 km):", grid["spatial_block"].nunique())
print("Saved:", GRID_GPKG)
print("Upload this to Earth Engine (zip all files):", up / "WB_grid_5km.shp")