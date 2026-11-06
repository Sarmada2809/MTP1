# West Bengal wealth and poverty mapping (5 km grid)

Replicates Shao & Li (2023) for West Bengal, plus extensions (extra features, spatial CV, baselines).

## Setup
    pip install pandas numpy geopandas shapely rasterio scikit-learn scipy matplotlib joblib openpyxl
    pip install libpysal esda        # optional, for Moran's I

## Put these in data_raw/
- West_Bengal_Clusters.xlsx  (from your WestBengal_Cluster_Data.py)
- DISTRICT_BOUNDARY.shp + .dbf .shx .prj
- West_Bengal_VIIRS_NTL_2019.tif / 2020 / 2021

## Run order
    python 01_build_grid.py
    python 02_attach_labels.py
    -> upload data_processed/gee_upload/WB_grid_5km.* to Earth Engine, run 03_gee_extract_features.js,
       download WB_gee_means.csv and WB_gee_pop.csv from Drive into data_raw/
    python 04_extract_ntl.py
    python 05_merge_dataset.py
    python 06_train_wealth_model.py
    python 07_poverty_rate_and_districts.py     # first run creates the official-poverty template
    (fill data_raw/district_official_poverty.csv, then rerun 07)
    python 08_make_maps.py

Check STATE_POVERTY_PCT in config.py against the NITI Aayog report before presenting.
