"""
STEP 6 - Train the Random Forest wealth model (the paper's Part 1) and evaluate it honestly.

  * trains on LABELLED cells only (cells with an NFHS wealth score)
  * tunes hyper-parameters with GridSearchCV using SPATIAL cross-validation
    (whole 20 km blocks are held out together, so neighbours never leak)
  * 'experiment ladder': A) NTL only, B) NTL + land cover (= the paper), C) + extra features
  * baselines: predict-the-mean, linear regression on night lights
  * also runs the paper-style RANDOM CV on model B, to show how much random CV flatters the score
  * repeats 5-fold spatial CV 10 times and reports mean +- SD (fixes the 'small test set' worry)
  * fits the best model on all labelled cells and predicts EVERY populated cell
    (labelled cells get their out-of-fold prediction, so they are not memorised values)
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from config import *

df = pd.read_csv(MASTER_CSV)
lab = df[df["labelled"] & df["model_ok"]].reset_index(drop=True)
y = lab["wealth_score"].to_numpy(float)
groups = lab["spatial_block"].astype(str).to_numpy()
print("Training cells:", len(lab), "| spatial blocks:", len(np.unique(groups)))


def spatial_folds(groups, k, seed):
    rng = np.random.RandomState(seed)
    ub = np.unique(groups)
    rng.shuffle(ub)
    fold_of = {b: i % k for i, b in enumerate(ub)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def random_folds(n, k, seed):
    rng = np.random.RandomState(seed)
    f = rng.permutation(n) % k
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def run_cv(make_model, X, spatial=True):
    rows, oofs = [], []
    for r in range(REPEATS):
        folds = spatial_folds(groups, K_FOLDS, r) if spatial else random_folds(len(y), K_FOLDS, r)
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            m = make_model()
            m.fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
        oof = np.clip(oof, 0, 100)
        rows.append({"R2": r2_score(y, oof), "MAE": mean_absolute_error(y, oof),
                     "RMSE": np.sqrt(mean_squared_error(y, oof))})
        oofs.append(oof)
    return pd.DataFrame(rows), np.mean(oofs, axis=0)


def summarise(name, cvdf, n_features):
    return {"model": name, "n_features": n_features,
            "R2_mean": cvdf.R2.mean(), "R2_sd": cvdf.R2.std(),
            "MAE_mean": cvdf.MAE.mean(), "MAE_sd": cvdf.MAE.std(),
            "RMSE_mean": cvdf.RMSE.mean(), "RMSE_sd": cvdf.RMSE.std()}


results, store = [], {}

# ---- baselines ----
Xd = lab[["gali_2021"]]
cv_d, _ = run_cv(lambda: DummyRegressor(strategy="mean"), Xd)
results.append(summarise("baseline_mean_prediction", cv_d, 0))
Xl = pd.DataFrame({"log_gali_2021": np.log1p(lab["gali_2021"])})
cv_l, _ = run_cv(lambda: LinearRegression(), Xl)
results.append(summarise("baseline_linear_on_NTL", cv_l, 1))

# ---- experiment ladder with Random Forest ----
for name, cols in FEATURE_SETS.items():
    X = lab[cols]
    gs = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), PARAM_GRID,
                      scoring="neg_mean_absolute_error",
                      cv=spatial_folds(groups, K_FOLDS, 999), n_jobs=-1)
    gs.fit(X, y)
    best = gs.best_params_
    print(f"\n{name}: best params {best}")
    make = lambda b=best: RandomForestRegressor(random_state=42, n_jobs=-1, **b)
    cvdf, oof = run_cv(make, X, spatial=True)
    results.append(summarise(f"RF_{name}_spatialCV", cvdf, len(cols)))
    store[name] = {"params": best, "oof": oof, "cols": cols, "mae": cvdf.MAE.mean()}
    print(f"   spatial CV  R2 {cvdf.R2.mean():.3f} +- {cvdf.R2.std():.3f} | "
          f"MAE {cvdf.MAE.mean():.2f} +- {cvdf.MAE.std():.2f}")
    if name == "B_paper_ntl_landcover":
        cv_r, _ = run_cv(make, X, spatial=False)
        results.append(summarise("RF_B_paper_RANDOM_CV (leaky, paper-style)", cv_r, len(cols)))
        print(f"   random CV   R2 {cv_r.R2.mean():.3f} | MAE {cv_r.MAE.mean():.2f}  <- optimistic")

res = pd.DataFrame(results).round(4)
res.to_csv(OUT / "experiment_ladder_results.csv", index=False)
print("\n=== EXPERIMENT LADDER ===")
print(res.to_string(index=False))

# ---- choose the best RF feature set (lowest spatial-CV MAE) ----
best_name = min(store, key=lambda k: store[k]["mae"])
b = store[best_name]
print("\nBest feature set:", best_name)

model = RandomForestRegressor(random_state=42, n_jobs=-1, **b["params"])
model.fit(lab[b["cols"]], y)
joblib.dump({"model": model, "features": b["cols"], "set": best_name,
             "params": b["params"]}, OUT / "random_forest_wealth_model.joblib")
json.dump({"best_set": best_name, "params": b["params"]},
          open(OUT / "model_summary.json", "w"), indent=2, default=str)

# ---- feature importance (like Fig. 4 in the paper) ----
imp = pd.DataFrame({"feature": b["cols"], "importance": model.feature_importances_})
imp["group"] = np.where(imp.feature.str.startswith(("gali", "gtli", "ntl_")), "Night lights",
                np.where(imp.feature.str.startswith("lc_"), "Land cover", "Other"))
imp.sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
print("\nImportance by group:\n", imp.groupby("group")["importance"].sum().round(3))

# ---- predictions for every modelled cell ----
mk = df[df["model_ok"]].copy()
mk["pred_full_model"] = np.clip(model.predict(mk[b["cols"]]), 0, 100)
oof_map = pd.Series(b["oof"], index=lab["Grid_ID"])
mk["pred_oof"] = mk["Grid_ID"].map(oof_map)
mk["pred_wealth"] = mk["pred_oof"].fillna(mk["pred_full_model"])
mk["pred_source"] = np.where(mk["pred_oof"].notna(), "out_of_fold", "full_model")
keep = ["Grid_ID", "spatial_block", "cx", "cy", "area_km2", "labelled", "wealth_score",
        "pop_sum", "pred_wealth", "pred_source", "pred_oof", "pred_full_model"]
mk[keep].to_csv(PRED_CSV, index=False)

oofdf = pd.DataFrame({"Grid_ID": lab["Grid_ID"], "actual": y, "pred_oof": b["oof"]})
oofdf["residual"] = oofdf["actual"] - oofdf["pred_oof"]
oofdf.to_csv(OOF_CSV, index=False)

# ---- Moran's I of residuals (are errors spatially clustered?) ----
try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    w = KNN.from_array(lab[["cx", "cy"]].to_numpy(), k=8)
    mi = Moran(oofdf["residual"].to_numpy(), w, permutations=499)
    print(f"\nMoran's I of residuals: {mi.I:.3f} (p = {mi.p_sim:.3f})")
except Exception as e:
    print("\n(Moran's I skipped - run: pip install libpysal esda)  ", type(e).__name__)

print("\nSaved predictions for", len(mk), "cells ->", PRED_CSV)
