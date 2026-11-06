"""
STEP 8 - Figures for the report/viva:
  1 labelled-cell map  2 predicted wealth map  3 poor/non-poor cells
  4 district poverty (estimated, and official if filled)  5 actual-vs-predicted  6 error histogram
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from config import *

grid = gpd.read_file(GRID_GPKG, layer="grid")[["Grid_ID", "geometry"]]
dist = gpd.read_file(DISTRICT_GPKG, layer="districts")
pv = pd.read_csv(POVERTY_CSV)
dc = pd.read_csv(OUT / "district_poverty_comparison.csv", dtype={"District": str})
oof = pd.read_csv(OOF_CSV)
gm = grid.merge(pv, on="Grid_ID", how="left")


def finish(ax, title):
    dist.boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_axis_off()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=250)
    plt.close(fig)
    print("saved", name)


fig, ax = plt.subplots(figsize=(6, 8))
gm.plot(ax=ax, color="#eeeeee", linewidth=0)
gm[gm["labelled"] == True].plot(ax=ax, color="crimson", linewidth=0)
finish(ax, "Cells with NFHS wealth labels (red)")
save(fig, "map1_labelled_cells.png")

fig, ax = plt.subplots(figsize=(6.5, 8))
gm.plot(column="pred_wealth", cmap="YlOrBr", legend=True, ax=ax, linewidth=0,
        missing_kwds={"color": "#eeeeee"}, legend_kwds={"label": "Predicted wealth (0-100)", "shrink": 0.6})
finish(ax, "Predicted wealth, 5 km cells")
save(fig, "map2_predicted_wealth.png")

fig, ax = plt.subplots(figsize=(6, 8))
gm.plot(ax=ax, color="#eeeeee", linewidth=0)
gm[gm["poor"] == True].plot(ax=ax, color="firebrick", linewidth=0)
finish(ax, f"Cells classed poor (state rate {STATE_POVERTY_PCT}%)")
save(fig, "map3_poor_cells.png")

dm = dist.merge(dc[["District", "est_poverty_pct"] +
                   (["official_poverty_pct"] if "official_poverty_pct" in dc else [])],
                on="District", how="left")
cols = ["est_poverty_pct"] + (["official_poverty_pct"]
        if "official_poverty_pct" in dm and dm["official_poverty_pct"].notna().any() else [])
vmax = float(dm[cols].max().max())
fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 8))
axes = np.atleast_1d(axes)
for ax, c in zip(axes, cols):
    dm.plot(column=c, cmap="Reds", vmin=0, vmax=vmax, legend=True, ax=ax,
            edgecolor="black", linewidth=0.4, legend_kwds={"shrink": 0.6, "label": "%"},
            missing_kwds={"color": "#dddddd"})
    ax.set_title("Estimated poverty rate (%)" if c == "est_poverty_pct" else "Official poverty rate (%)")
    ax.set_axis_off()
save(fig, "map4_district_poverty.png")

r2 = r2_score(oof["actual"], oof["pred_oof"])
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(oof["actual"], oof["pred_oof"], alpha=0.6, s=18)
ax.plot([0, 100], [0, 100], "r--")
ax.set_xlabel("Actual NFHS wealth score"); ax.set_ylabel("Predicted (out-of-fold)")
ax.set_title(f"Spatial-CV predictions (R2 = {r2:.3f})"); ax.grid(alpha=0.3)
save(fig, "fig5_actual_vs_predicted.png")

fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(oof["residual"].abs(), bins=25, edgecolor="black")
ax.set_xlabel("Absolute error"); ax.set_ylabel("Cells"); ax.grid(alpha=0.3)
ax.set_title("Absolute prediction errors (out-of-fold)")
save(fig, "fig6_error_histogram.png")