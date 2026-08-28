#!/usr/bin/env python3
"""Recompute the manuscript quantities that can be verified from this repository."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ROOT = HERE.parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

DATA = ROOT / "data" / "derived" / "data_with_descriptors.csv"
RAW_RESULTS = ROOT / "results"


def pooled_cv(estimator_factory, X, y, splitter, groups=None):
    predictions = np.full(y.shape, np.nan, dtype=float)
    split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for train, test in split_iter:
        estimator = estimator_factory()
        estimator.fit(X[train], y[train])
        predictions[test] = estimator.predict(X[test])
    return {
        "R2": r2_score(y, predictions),
        "RMSE": mean_squared_error(y, predictions) ** 0.5,
        "n": len(y),
    }


def pca_ols_audit(df):
    features = [
        "VEC", "dH_mix", "dS_mix", "Omega", "delta_chi", "delta",
        "d_inv_sqrt", "ColdWork", "RecrystT", "HoldTime", "SD_GS",
    ]
    use = df.dropna(subset=["YS", *features]).reset_index(drop=True)
    X = use[features].to_numpy(float)
    y = use["YS"].to_numpy(float)
    groups = use["Iteration"].to_numpy()

    def estimator():
        return Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=6)),
            ("ols", LinearRegression()),
        ])

    protocols = {
        "5-fold": (KFold(n_splits=5, shuffle=True, random_state=42), None),
        "LOO": (LeaveOneOut(), None),
        "LOBO": (LeaveOneGroupOut(), groups),
    }
    rows = []
    for protocol, (splitter, split_groups) in protocols.items():
        metrics = pooled_cv(estimator, X, y, splitter, split_groups)
        rows.append({
            "model": "PCA-OLS",
            "protocol": protocol,
            "preprocessing": "StandardScaler and PCA fit within every training fold",
            "features": "+".join(features),
            **metrics,
        })
    pd.DataFrame(rows).to_csv(OUT / "pca_ols_nested_results.csv", index=False)


def linear_form_factory():
    return LinearRegression(fit_intercept=False)


def hv_equation_audit(df):
    use = df.dropna(subset=["HV", "GrainSize", "SD_GS", "dH_mix", "HoldTime"]).reset_index(drop=True)
    d = use["GrainSize"].to_numpy(float)
    sd = use["SD_GS"].to_numpy(float)
    dh = use["dH_mix"].to_numpy(float)
    hold = use["HoldTime"].to_numpy(float)
    y = use["HV"].to_numpy(float)
    groups = use["Iteration"].to_numpy()
    X = np.column_stack([np.ones(len(use)), (6.93 - d) / sd, dh / hold**2])

    full = linear_form_factory().fit(X, y)
    full_pred = full.predict(X)
    rows = [{
        "model": "HV corrected fixed form",
        "protocol": "full fit",
        "R2": r2_score(y, full_pred),
        "RMSE": mean_squared_error(y, full_pred) ** 0.5,
        "c0": full.coef_[0],
        "c1": full.coef_[1],
        "c2": full.coef_[2],
        "equation": "HV = c0 + c1*(6.93-d)/SD_GS + c2*dH_mix/HoldTime^2",
        "status": "form selected previously; coefficients fit to all data",
    }]

    protocols = {
        "5-fold": (KFold(n_splits=5, shuffle=True, random_state=42), None),
        "LOO": (LeaveOneOut(), None),
        "LOBO": (LeaveOneGroupOut(), groups),
    }
    for protocol, (splitter, split_groups) in protocols.items():
        metrics = pooled_cv(linear_form_factory, X, y, splitter, split_groups)
        rows.append({
            "model": "HV corrected fixed form",
            "protocol": protocol,
            **metrics,
            "c0": np.nan,
            "c1": np.nan,
            "c2": np.nan,
            "equation": "coefficients refit within every training fold",
            "status": "post-selection fixed-form validation; not nested equation discovery",
        })
    pd.DataFrame(rows).to_csv(OUT / "hardness_equation_corrected.csv", index=False)

    rng = np.random.default_rng(42)
    boot = []
    for _ in range(10_000):
        idx = rng.integers(0, len(y), len(y))
        boot.append(linear_form_factory().fit(X[idx], y[idx]).coef_)
    boot = np.asarray(boot)
    constants = []
    for index, name in enumerate(["c0", "c1", "c2"]):
        constants.append({
            "constant": name,
            "estimate": full.coef_[index],
            "bootstrap_mean": boot[:, index].mean(),
            "bootstrap_sd": boot[:, index].std(ddof=1),
            "ci_2.5": np.quantile(boot[:, index], 0.025),
            "ci_97.5": np.quantile(boot[:, index], 0.975),
        })
    pd.DataFrame(constants).to_csv(OUT / "hardness_equation_bootstrap.csv", index=False)


def dataset_audit(df):
    comp_cols = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "V"]
    comp_key = df[comp_cols].round(8).astype(str).agg("|".join, axis=1)
    grouped = df.assign(_comp=comp_key).groupby("_comp", sort=False)
    repeated = grouped.size()
    cross_batch = grouped["Iteration"].nunique()
    cross_keys = cross_batch[cross_batch > 1].index
    cross_rows = df.assign(_comp=comp_key).loc[comp_key.isin(cross_keys), ["Iteration", "Alloy", *comp_cols]]
    cross_rows.to_csv(OUT / "cross_batch_repeated_compositions.csv", index=False)

    c_campaign = df[df["Iteration"].str.startswith("C")]
    ydf = df.dropna(subset=["YS"])
    y_comp_key = ydf[comp_cols].round(8).astype(str).agg("|".join, axis=1)
    summary = pd.DataFrame([{
        "n_total": len(df),
        "n_YS": ydf.shape[0],
        "n_HV": df["HV"].notna().sum(),
        "n_unique_compositions_all": repeated.shape[0],
        "n_unique_compositions_YS": y_comp_key.nunique(),
        "n_repeated_composition_groups": int((repeated > 1).sum()),
        "n_rows_in_repeated_groups": int(repeated[repeated > 1].sum()),
        "n_compositions_crossing_batches": len(cross_keys),
        "C_campaign_hold_min_h": c_campaign["HoldTime"].min(),
        "C_campaign_hold_max_h": c_campaign["HoldTime"].max(),
        "corr_mean_d_SD_GS": df[["GrainSize", "SD_GS"]].corr().iloc[0, 1],
        "YS_min_MPa": ydf["YS"].min(),
        "YS_max_MPa": ydf["YS"].max(),
        "HV_min": df["HV"].min(),
        "HV_max": df["HV"].max(),
        "d_min_um": df["GrainSize"].min(),
        "d_max_um": df["GrainSize"].max(),
    }])
    summary.to_csv(OUT / "dataset_audit.csv", index=False)


def external_audit():
    ext = pd.read_csv(RAW_RESULTS / "external_tier_results.csv")
    ext["evidence_tier"] = ext["label"].str.extract(r"(Tier \d|Aggregate)")[0]
    ext["target_status"] = np.select(
        [
            ext["label"].str.contains("measured YS"),
            ext["label"].str.contains("HP-derived"),
            ext["label"].str.contains("HV-converted"),
        ],
        ["direct YS measurement", "YS reconstructed from published Hall-Petch fit", "YS proxy converted from HV"],
        default="mixed aggregate",
    )
    ext["beats_mean_baseline"] = ext["R2"] > 0
    ext.to_csv(OUT / "external_validation_by_evidence_tier.csv", index=False)


def method_audit():
    rows = [
        {
            "result_family": "PySR grid",
            "current_evidence": "full-data equation search followed by fold-wise constant refit",
            "allowed_claim": "post-selection fixed-form CV",
            "prohibited_claim": "nested out-of-sample symbolic discovery",
        },
        {
            "result_family": "M15",
            "current_evidence": "cached elpd only; source streams absent",
            "allowed_claim": "provisional archival result",
            "prohibited_claim": "fully reproducible headline model",
        },
        {
            "result_family": "PCA-OLS",
            "current_evidence": "recomputed with scaling and PCA inside every fold",
            "allowed_claim": "fold-contained PCA baseline",
            "prohibited_claim": "descriptor-only model; processing variables are included",
        },
        {
            "result_family": "External YS",
            "current_evidence": "54 measured, 3 HP-derived, 25 HV-converted targets",
            "allowed_claim": "tiered external stress test",
            "prohibited_claim": "82 independent measured YS observations or demonstrated deployment",
        },
        {
            "result_family": "Singularity audit",
            "current_evidence": "observed-range checks for four registered equations",
            "allowed_claim": "targeted domain and denominator audit",
            "prohibited_claim": "convex-hull sweep of every discovered equation",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "method_claim_audit.csv", index=False)


def main():
    df = pd.read_csv(DATA)
    dataset_audit(df)
    pca_ols_audit(df)
    hv_equation_audit(df)
    external_audit()
    method_audit()
    print(f"Verified outputs written to {OUT}")


if __name__ == "__main__":
    main()
