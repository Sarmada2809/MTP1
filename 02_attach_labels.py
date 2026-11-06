"""
STEP 2 - Attach NFHS wealth labels to the grid.
  * puts each NFHS cluster into the 5 km cell that contains it
  * cell wealth = HOUSEHOLD-WEIGHTED mean of the cluster scores (big clusters count more)
  * cells with at least one cluster become 'labelled'; all other cells stay empty
    and will be predicted later
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from config import *

cl = pd.read_excel(CLUSTER_XLSX)
cl.columns = cl.columns.str.strip()
need = ["Cluster number", "Latitude", "Longitude",
        "Normalized wealth score", "Household count"]
miss = [c for c in need if c not in cl.columns]
if miss:
    raise ValueError(f"Missing columns in cluster file: {miss}")
for c in need:
    cl[c] = pd.to_numeric(cl[c], errors="coerce")
cl = cl.dropna(subset=need)
cl = cl[cl["Household count"] > 0].copy()

pts = gpd.GeoDataFrame(cl, geometry=gpd.points_from_xy(cl["Longitude"], cl["Latitude"]),
                       crs=CRS_GEO).to_crs(CRS_PROJ)
G = GRID_SIZE_M
pts["gx"] = np.floor(pts.geometry.x / G).astype(int)
pts["gy"] = np.floor(pts.geometry.y / G).astype(int)
pts["Grid_ID"] = pts["gx"].astype(str) + "_" + pts["gy"].astype(str)

grid = gpd.read_file(GRID_GPKG, layer="grid")
valid = set(grid["Grid_ID"])
inside = pts["Grid_ID"].isin(valid)
print("Clusters read:", len(pts))
print("Clusters falling outside the grid (dropped):", int((~inside).sum()))
pts = pts[inside].copy()

pts["w"] = pts["Household count"]
pts["w_score"] = pts["w"] * pts["Normalized wealth score"]
lab = pts.groupby("Grid_ID").agg(
    cluster_count=("Cluster number", "nunique"),
    household_count=("w", "sum"),
    w_score=("w_score", "sum"),
    wealth_mean_unweighted=("Normalized wealth score", "mean"),
).reset_index()
lab["wealth_score"] = lab["w_score"] / lab["household_count"]
lab = lab.drop(columns="w_score")

lab.to_csv(LABELS_CSV, index=False)
pts.drop(columns="geometry").to_csv(CLUSTER_ASSIGN_CSV, index=False)

print("\nTotal grid cells:", len(grid))
print("Labelled cells:", len(lab), f"({len(lab) / len(grid) * 100:.1f}% of all cells)")
print("Clusters per labelled cell (mean):", round(lab["cluster_count"].mean(), 2))
print(lab["wealth_score"].describe().round(2))
print("Saved:", LABELS_CSV)
