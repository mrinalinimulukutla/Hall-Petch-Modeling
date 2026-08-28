#!/usr/bin/env python3
"""Pooled grouped-validation table (Table 3 of the manuscript).

Writes results/main2_grouped_validation.csv: 5-fold / LOO / LOBO pooled Q^2
for the headline model of each family, for both YS and HV.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", message="X does not have valid feature names")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "derived" / "data_with_descriptors.csv"
OUT = ROOT / "results" / "main2_grouped_validation.csv"
SEED = 42

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "V"]
NON_NI = [element for element in ELEMENTS if element != "Ni"]
WEN = ["VEC", "dH_mix", "dS_mix", "Omega", "delta_chi", "delta"]
PCA_FEATURES = WEN + [
    "d_inv_sqrt", "SD_GS", "ColdWork", "RecrystT", "HoldTime"
]


def make_estimator(name):
    if name in {
        "F1 classical Hall-Petch",
        "F3 M3",
        "F3 M15 SD-grain interaction",
        "F5 HV fixed form",
    }:
        return LinearRegression()
    if name == "F2 PCA-OLS":
        return Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=6)),
            ("model", LinearRegression()),
        ])
    if name == "F4 Lasso S2":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=1.0, max_iter=10_000, random_state=SEED)),
        ])
    if name == "F4 LightGBM S2":
        return LGBMRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "F4 LightGBM S1":
        return LGBMRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(name)


def design_matrix(frame, model):
    if model == "F1 classical Hall-Petch":
        return frame[["d_inv_sqrt"]].to_numpy(float)
    if model == "F3 M3":
        return frame[[f"{element}_frac" for element in NON_NI] + ["d_inv_sqrt"]].to_numpy(float)
    if model == "F3 M15 SD-grain interaction":
        base = frame[[f"{element}_frac" for element in NON_NI] + ["d_inv_sqrt", "SD_GS"]].to_numpy(float)
        interaction = (frame["SD_GS"] * frame["d_inv_sqrt"]).to_numpy(float)
        return np.column_stack([base, interaction])
    if model == "F2 PCA-OLS":
        return frame[PCA_FEATURES].to_numpy(float)
    if model in {"F4 Lasso S2", "F4 LightGBM S2"}:
        return frame[["d_inv_sqrt", "SD_GS", *WEN]].to_numpy(float)
    if model == "F4 LightGBM S1":
        return frame[["d_inv_sqrt", "SD_GS"]].to_numpy(float)
    if model == "F5 HV fixed form":
        return np.column_stack([
            (6.93 - frame["GrainSize"].to_numpy(float)) / frame["SD_GS"].to_numpy(float),
            frame["dH_mix"].to_numpy(float) / frame["HoldTime"].to_numpy(float) ** 2,
        ])
    raise ValueError(model)


def out_of_fold_predictions(model_name, x, y, splitter, groups=None):
    predictions = np.full(len(y), np.nan)
    split_iter = (
        splitter.split(x, y, groups)
        if groups is not None
        else splitter.split(x, y)
    )
    for train, test in split_iter:
        estimator = clone(make_estimator(model_name))
        estimator.fit(x[train], y[train])
        predictions[test] = estimator.predict(x[test])
    return predictions


def batch_cluster_interval(y, predictions, batches, n_boot=1000):
    unique_batches = np.unique(batches)
    indices = {batch: np.flatnonzero(batches == batch) for batch in unique_batches}
    rng = np.random.default_rng(SEED)
    boot_r2 = []
    boot_rmse = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_batches, size=len(unique_batches), replace=True)
        take = np.concatenate([indices[batch] for batch in sampled])
        if np.var(y[take]) == 0:
            continue
        boot_r2.append(r2_score(y[take], predictions[take]))
        boot_rmse.append(mean_squared_error(y[take], predictions[take]) ** 0.5)
    return (
        np.quantile(boot_r2, [0.025, 0.975]),
        np.quantile(boot_rmse, [0.025, 0.975]),
    )


def evaluate(frame, target, model_name):
    required = [target, "Iteration", *ELEMENTS]
    use = frame.dropna(subset=required).reset_index(drop=True)
    x = design_matrix(use, model_name)
    y = use[target].to_numpy(float)
    batches = use["Iteration"].to_numpy()
    protocols = {
        "5-fold": (KFold(n_splits=5, shuffle=True, random_state=SEED), None),
        "LOO": (LeaveOneOut(), None),
        "LOBO": (LeaveOneGroupOut(), batches),
    }
    rows = []
    for protocol, (splitter, groups) in protocols.items():
        predictions = out_of_fold_predictions(model_name, x, y, splitter, groups)
        r2_ci, rmse_ci = batch_cluster_interval(y, predictions, batches)
        rows.append({
            "target": target,
            "model": model_name,
            "protocol": protocol,
            "n": len(y),
            "R2": r2_score(y, predictions),
            "R2_batch_boot_low": r2_ci[0],
            "R2_batch_boot_high": r2_ci[1],
            "RMSE": mean_squared_error(y, predictions) ** 0.5,
            "RMSE_batch_boot_low": rmse_ci[0],
            "RMSE_batch_boot_high": rmse_ci[1],
        })
    return rows


def main():
    frame = pd.read_csv(DATA)
    specs = [
        ("YS", "F1 classical Hall-Petch"),
        ("YS", "F2 PCA-OLS"),
        ("YS", "F3 M3"),
        ("YS", "F3 M15 SD-grain interaction"),
        ("YS", "F4 Lasso S2"),
        ("YS", "F4 LightGBM S2"),
        ("HV", "F1 classical Hall-Petch"),
        ("HV", "F4 LightGBM S1"),
        ("HV", "F5 HV fixed form"),
    ]
    rows = []
    for target, model in specs:
        rows.extend(evaluate(frame, target, model))
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
