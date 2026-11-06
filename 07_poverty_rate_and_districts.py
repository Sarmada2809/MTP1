"""
STEP 7 - From wealth map to poverty rate (the paper's Part 2).

 1. Sort all populated cells from poorest to richest predicted wealth.
 2. Walk up the list adding WorldPop population until the poor share equals the
    OFFICIAL state poverty rate. The wealth value reached is the CUTOFF.
 3. Cells with predicted wealth <= cutoff are 'poor cells'.
 4. Split every cell's population across the districts it overlaps (by area share),
    then district poverty rate = poor population / total population.
 5. Compare with official district poverty rates (data_raw/district_official_poverty.csv).
"""
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from config import *

pred = pd.read_csv(PRED_CSV)
grid = gpd.read_file(GRID_GPKG, layer="grid")[["Grid_ID", "geometry"]]
dist = gpd.read_file(DISTRICT_GPKG, layer="districts")

p = STATE_POVERTY_PCT / 100.0
d = pred[pred["pop_sum"] > 0].sort_values("pred_wealth").reset_index(drop=True)
cum = (d["pop_sum"].cumsum() / d["pop_sum"].sum()).to_numpy()
idx = min(int(np.searchsorted(cum, p)), len(d) - 1)
cutoff = float(d.loc[idx, "pred_wealth"])
print(f"State poverty rate used: {STATE_POVERTY_PCT}%")
print(f"Wealth cutoff (0-100 scale): {cutoff:.2f}")

pred["poor"] = (pred["pred_wealth"] <= cutoff) & (pred["pop_sum"] > 0)
pred.to_csv(POVERTY_CSV, index=False)

# ---- split population among districts ----
cells = grid.merge(pred[["Grid_ID", "pop_sum", "poor"]], on="Grid_ID")
inter = gpd.overlay(cells, dist[["District", "geometry"]], how="intersection",
                    keep_geom_type=True)
inter["piece_area"] = inter.geometry.area
inter["frac"] = inter["piece_area"] / inter.groupby("Grid_ID")["piece_area"].transform("sum")
inter["pop_piece"] = inter["pop_sum"] * inter["frac"]
inter["poor_piece"] = inter["pop_piece"] * inter["poor"].astype(float)

dt = inter.groupby("District").agg(population=("pop_piece", "sum"),
                                   poor_population=("poor_piece", "sum")).reset_index()
dt["est_poverty_pct"] = dt["poor_population"] / dt["population"] * 100
print("\nEstimated state poverty rate (check):",
      round(dt["poor_population"].sum() / dt["population"].sum() * 100, 2), "%")

# ---- compare with official numbers ----
def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())

dt["key"] = dt["District"].map(norm)
if not OFFICIAL_CSV.exists():
    tmpl = pd.DataFrame({"District": dt["District"], "official_poverty_pct": np.nan})
    tmpl.to_csv(OFFICIAL_CSV, index=False)
    print(f"\nCreated template {OFFICIAL_CSV}\nFill 'official_poverty_pct' from the NITI Aayog "
          "district MPI headcount ratios, then run this script again.")
else:
    off = pd.read_csv(OFFICIAL_CSV, dtype={"District": str})
    off["key"] = off["District"].map(norm)
    off = off.rename(columns={"District": "District_official"})
    dt = dt.merge(off[["key", "District_official", "official_poverty_pct"]], on="key", how="left")
    ok = dt.dropna(subset=["official_poverty_pct"])
    if len(ok) >= 3:
        a, e = ok["official_poverty_pct"].to_numpy(), ok["est_poverty_pct"].to_numpy()
        rel = np.mean(np.abs(e - a) / np.maximum(a, 1e-6)) * 100
        print(f"\nDistricts compared: {len(ok)} of {len(dt)}")
        print(f"R2 (district): {r2_score(a, e):.3f}")
        print(f"Spearman rank correlation: {spearmanr(a, e)[0]:.3f}")
        print(f"MAE (percentage points): {mean_absolute_error(a, e):.2f}")
        print(f"Mean relative error: {rel:.1f}%  -> 'accuracy' {100 - rel:.1f}%")
        print("Unmatched names:", sorted(set(dt.loc[dt.official_poverty_pct.isna(), "District"])))
    else:
        print("\nFill official_poverty_pct in", OFFICIAL_CSV, "(need at least 3 matching districts).")

dt.drop(columns="key").round(3).to_csv(OUT / "district_poverty_comparison.csv", index=False)
print("Saved:", OUT / "district_poverty_comparison.csv")