"""
STEP 1b - Match shapefile district NAMES to NFHS district names (run before 01_build_grid.py).

How:
  1. take each shapefile polygon's name (column 'District') and normalise spelling
     (case, spaces, 'Hooghly'->'Hugli', 'Howrah'->'Haora', '24'->'twentyfour', ...)
  2. recent new districts are merged into the NFHS-5 district they were carved from:
        Alipurduar -> Jalpaiguri, Kalimpong -> Darjiling, Jhargram -> Paschim Medinipur
  3. match to the NFHS names found in your cluster file (which come from District.txt)
  4. CROSS-CHECK: each NFHS cluster is placed in its polygon and votes with its NFHS name.
     Any row where the name match and the vote disagree is flagged.
Output: data_raw/district_name_map.csv  (edit 'final_name' if anything is wrong)
"""
import re
import pandas as pd
import geopandas as gpd
from config import *

ALIAS = {   # normalised variant -> normalised NFHS spelling
    "hooghly": "hugli", "howrah": "haora", "coochbehar": "kochbihar",
    "purulia": "puruliya", "malda": "maldah", "darjeeling": "darjiling",
    "dakshindinajpur": "dakshindinajpur", "southdinajpur": "dakshindinajpur",
    "northdinajpur": "uttardinajpur", "westmidnapore": "paschimmedinipur",
    "eastmidnapore": "purbamedinipur", "paschimmidnapore": "paschimmedinipur",
    "purbamidnapore": "purbamedinipur", "pashchimmedinipur": "paschimmedinipur",
    "pashchimbarddhaman": "paschimbarddhaman", "purbabarddhaman": "purbabarddhaman",
    "paschimbardhaman": "paschimbarddhaman", "purbabardhaman": "purbabarddhaman",
    "pashchimbardhaman": "paschimbarddhaman",
    "north24parganas": "northtwentyfourparganas",
    "south24parganas": "southtwentyfourparganas",
}
NEW_TO_PARENT = {"alipurduar": "jalpaiguri", "kalimpong": "darjiling",
                 "jhargram": "paschimmedinipur"}


def norm(s):
    s = str(s).lower().replace("24", "twentyfour")
    s = re.sub(r"[^a-z]", "", s)
    return s


raw = gpd.read_file(DISTRICT_SHP)
key = raw[STATE_FIELD].astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
wb = raw.loc[key == "westbengal", [DISTRICT_FIELD, "geometry"]].copy()
wb["poly_name"] = wb[DISTRICT_FIELD].astype(str).str.strip()
wb = wb.to_crs(CRS_PROJ).dissolve(by="poly_name", as_index=False)[["poly_name", "geometry"]]
print("West Bengal polygons:", len(wb))
print(sorted(wb["poly_name"]), "\n")

cl = pd.read_excel(CLUSTER_XLSX)
cl.columns = cl.columns.str.strip()
name_col = "District name from file"
cl = cl.dropna(subset=[name_col, "Latitude", "Longitude"])
nfhs_names = sorted(cl[name_col].astype(str).str.strip().unique())
nfhs_by_key = {norm(n): n for n in nfhs_names}
print("NFHS districts in your cluster file (%d):" % len(nfhs_names), nfhs_names, "\n")

# cross-check by location
pts = gpd.GeoDataFrame(cl, geometry=gpd.points_from_xy(cl.Longitude, cl.Latitude),
                       crs=CRS_GEO).to_crs(CRS_PROJ)
j = gpd.sjoin_nearest(pts[[name_col, "geometry"]], wb, how="left", max_distance=5000)

rows = []
for pn in sorted(wb["poly_name"]):
    k = norm(pn)
    k = ALIAS.get(k, k)
    k = NEW_TO_PARENT.get(k, k)
    by_name = nfhs_by_key.get(k)
    vc = j[j["poly_name"] == pn][name_col].astype(str).str.strip().value_counts()
    by_vote = vc.index[0] if len(vc) else None
    final = by_name or by_vote or "UNKNOWN"
    rows.append({
        "shapefile_code": pn,                       # (kept column name; holds the polygon NAME)
        "final_name": final,
        "matched_by": "name" if by_name else ("cluster_vote" if by_vote else "none"),
        "name_match": by_name, "vote_match": by_vote, "n_clusters": int(vc.sum()),
        "agree": (by_name == by_vote) if (by_name and by_vote) else None,
    })
m = pd.DataFrame(rows)
m.to_csv(DATA_RAW / "district_name_map.csv", index=False)
print(m.to_string(index=False))

bad = m[m["agree"] == False]
if len(bad):
    print("\nNAME vs VOTE DISAGREE (check these):\n", bad[["shapefile_code", "name_match", "vote_match"]].to_string(index=False))
none = m[m["matched_by"] != "name"]
if len(none):
    print("\nNot matched by name (used vote or UNKNOWN):", list(none["shapefile_code"]))
missing = sorted(set(nfhs_names) - set(m["final_name"]))
if missing:
    print("\nNFHS districts with NO polygon:", missing)
print("\nDistricts after merging:", m["final_name"].nunique())
print("Saved:", DATA_RAW / "district_name_map.csv")
