"""
STEP 5 - Merge everything into ONE table (one row per 5 km cell).
    grid info + wealth label + NTL features + GEE features (land cover, population, ...)
Also decides which cells are 'populated' (the cells we model and map).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from config import *

g = gpd.read_file(GRID_GPKG, layer="grid")
cent = g.geometry.centroid
g["cx"], g["cy"] = cent.x, cent.y
df = pd.DataFrame(g.drop(columns="geometry"))

labels = pd.read_csv(LABELS_CSV)
ntl = pd.read_csv(NTL_CSV)
means = pd.read_csv(GEE_MEANS_CSV)
pop = pd.read_csv(GEE_POP_CSV)
means = means[[c for c in means.columns if not c.startswith("system") and c != ".geo"]]
pop = pop[["Grid_ID", "pop_sum"]]

for name, t in [("labels", labels), ("ntl", ntl), ("gee_means", means), ("gee_pop", pop)]:
    if t["Grid_ID"].duplicated().any():
        raise ValueError(f"Duplicate Grid_ID in {name}")

df = (df.merge(labels, on="Grid_ID", how="left")
        .merge(ntl, on="Grid_ID", how="left")
        .merge(means, on="Grid_ID", how="left")
        .merge(pop, on="Grid_ID", how="left"))

# fill gaps sensibly
for c in LC_COLS + ["built_surface"]:
    df[c] = df[c].fillna(0)
for c in ["elevation", "ndvi"]:
    df[c] = df[c].fillna(df[c].median())
df["pop_sum"] = df["pop_sum"].fillna(0).clip(lower=0)
df["pop_density"] = df["pop_sum"] / df["area_km2"]
df["cluster_count"] = df["cluster_count"].fillna(0)
df["household_count"] = df["household_count"].fillna(0)

df["labelled"] = df["wealth_score"].notna()
ntl_ok = df[NTL_PAPER].notna().all(axis=1)
df["model_ok"] = ntl_ok & ((df["pop_sum"] >= POP_MIN) | df["labelled"])

lc_sum = df[LC_COLS].sum(axis=1)
print("Land-cover fractions sum (should be ~0.9-1.0):", round(lc_sum.mean(), 3))
df.to_csv(MASTER_CSV, index=False)

populated = int((df["pop_sum"] >= POP_MIN).sum())
print("\nAll cells:", len(df))
print("Populated cells (WorldPop >= %g):" % POP_MIN, populated)
print("Cells used for modelling/mapping:", int(df["model_ok"].sum()))
print("Labelled cells used for training:", int((df["labelled"] & df["model_ok"]).sum()))
print("Label coverage of populated cells: %.1f%%" %
      (df.loc[df.model_ok, "labelled"].mean() * 100))
print("Saved:", MASTER_CSV)
