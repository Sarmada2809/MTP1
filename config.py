"""
Shared settings. Every script imports this file, so change things HERE only.
Folder layout (everything lives inside WB_Poverty_Project):
    data_raw/        <- files you provide (NFHS clusters, shapefile, NTL tifs, GEE csvs)
    data_processed/  <- files the scripts create
    outputs/         <- final tables, models, maps
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data_raw"
DATA_PROC = ROOT / "data_processed"
OUT = ROOT / "outputs"
for _p in (DATA_RAW, DATA_PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------- grid settings ----------------
GRID_SIZE_M = 5000          # 5 km x 5 km cells
BLOCK_CELLS = 4             # 4 x 4 cells = 20 km spatial blocks (used for honest CV)
CRS_PROJ = "EPSG:32645"     # UTM 45N, metres (covers West Bengal)
CRS_GEO = "EPSG:4326"
MIN_FRAC_IN_STATE = 0.02    # drop sliver cells with <2% of their area inside West Bengal

# ---------------- input files (put these in data_raw/) ----------------
CLUSTER_XLSX = DATA_RAW / "West_Bengal_Clusters.xlsx"      # output of your WestBengal_Cluster_Data.py
DISTRICT_SHP = DATA_RAW / "DISTRICT_BOUNDARY.shp"          # keep .dbf .shx .prj next to it
DISTRICT_FIELD = "District"       # column with district NAMES in the shapefile
STATE_FIELD = "STATE"
NTL_TIFS = {y: DATA_RAW / f"West_Bengal_VIIRS_NTL_{y}.tif" for y in (2019, 2020, 2021)}
GEE_MEANS_CSV = DATA_RAW / "WB_gee_means.csv"              # exported by 03_gee_extract_features.js
GEE_POP_CSV = DATA_RAW / "WB_gee_pop.csv"                  # exported by 03_gee_extract_features.js
OFFICIAL_CSV = DATA_RAW / "district_official_poverty.csv"  # you fill this (template is auto-created)

# ---------------- files created by the scripts ----------------
GRID_GPKG = DATA_PROC / "wb_grid_5km.gpkg"
DISTRICT_GPKG = DATA_PROC / "wb_districts.gpkg"
LABELS_CSV = DATA_PROC / "grid_labels.csv"
CLUSTER_ASSIGN_CSV = DATA_PROC / "cluster_grid_assignments.csv"
NTL_CSV = DATA_PROC / "ntl_features.csv"
MASTER_CSV = DATA_PROC / "master_dataset.csv"
PRED_CSV = OUT / "wealth_predictions.csv"
OOF_CSV = OUT / "oof_predictions.csv"
POVERTY_CSV = OUT / "grid_poverty_status.csv"

# ---------------- poverty step ----------------
# Official state-level poverty headcount (%) used to calibrate the cutoff.
# NITI Aayog National MPI (NFHS-5 based) reports about 11.89 for West Bengal.
# VERIFY this number in the NITI Aayog report before you present.
STATE_POVERTY_PCT = 11.89
POP_MIN = 1.0               # a cell counts as "populated" if WorldPop >= this many people

# ---------------- feature groups ----------------
LC_COLS = ["lc_crop", "lc_tree", "lc_grass", "lc_shrub",
           "lc_wetland", "lc_built", "lc_bare", "lc_water"]
NTL_PAPER = ([f"gali_{y}" for y in (2019, 2020, 2021)]
             + [f"gtli_{y}" for y in (2019, 2020, 2021)])
EXTRA = ["ntl_max_2020", "ntl_std_2020", "pop_density",
         "built_surface", "elevation", "ndvi"]

FEATURE_SETS = {
    "A_ntl_only": NTL_PAPER,
    "B_paper_ntl_landcover": NTL_PAPER + LC_COLS,          # the paper's model
    "C_extended": NTL_PAPER + LC_COLS + EXTRA,             # our extension
}

# ---------------- modelling ----------------
K_FOLDS = 5
REPEATS = 10
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [6, 10, None],
    "min_samples_leaf": [2, 5],
    "max_features": [0.5, 1.0],
}