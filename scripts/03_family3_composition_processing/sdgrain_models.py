#!/usr/bin/env python3
"""Verify the incremental YS contribution of SD_GS and SD_GS*d^-1/2."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "derived" / "data_with_descriptors.csv"
OUT = ROOT / "results" / "sdgrain_m15_validation.csv"
OUT_FEATURE = ROOT / "results" / "sdgrain_feature_audit.csv"
ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "V"]
BASE = [f"{element}_frac" for element in ELEMENTS] + ["d_inv_sqrt"]


def pooled_predictions(x, y, splitter, groups=None):
    predictions = np.empty(len(y))
    folds = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
    for train, test in folds:
        model = LinearRegression().fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
    return predictions


def direction_invariant_auc(labels, values):
    auc = roc_auc_score(labels, values)
    return max(auc, 1.0 - auc)


def main():
    frame = pd.read_csv(DATA).dropna(subset=["YS"]).reset_index(drop=True)
    frame["SD_x_dinv"] = frame["SD_GS"] * frame["d_inv_sqrt"]
    designs = {
        "M3": BASE,
        "M3 + additive SD_grain": BASE + ["SD_GS"],
        "M15: M3 + SD_grain + SD_grain*d^-1/2": BASE + ["SD_GS", "SD_x_dinv"],
    }
    splitters = {
        "5-fold": (KFold(n_splits=5, shuffle=True, random_state=42), None),
        "LOO": (LeaveOneOut(), None),
        "LOBO": (LeaveOneGroupOut(), frame["Iteration"].to_numpy()),
    }
    y = frame["YS"].to_numpy(float)
    rows = []
    for name, columns in designs.items():
        x = frame[columns].to_numpy(float)
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        for protocol, (splitter, groups) in splitters.items():
            predictions = pooled_predictions(x, y, splitter, groups)
            rows.append({
                "model": name,
                "protocol": protocol,
                "n": len(y),
                "n_parameters": x.shape[1] + 1,
                "train_R2": fit.rsquared,
                "BIC": fit.bic,
                "R2": r2_score(y, predictions),
                "RMSE_MPa": mean_squared_error(y, predictions) ** 0.5,
                "interaction_coefficient": fit.params[-1] if name.startswith("M15") else np.nan,
                "interaction_t": fit.tvalues[-1] if name.startswith("M15") else np.nan,
                "interaction_p": fit.pvalues[-1] if name.startswith("M15") else np.nan,
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {OUT}")

    all_records = pd.read_csv(DATA).reset_index(drop=True)
    feature_rows = []
    for target in ("YS", "HV"):
        use = all_records.dropna(subset=[target]).reset_index(drop=True)
        target_y = use[target].to_numpy(float)
        target_groups = use["Iteration"].to_numpy()
        for name, columns in {
            "d^-1/2 only": ["d_inv_sqrt"],
            "d^-1/2 + additive SD_grain": ["d_inv_sqrt", "SD_GS"],
        }.items():
            target_x = use[columns].to_numpy(float)
            for protocol, (splitter, groups) in {
                "LOO": (LeaveOneOut(), None),
                "LOBO": (LeaveOneGroupOut(), target_groups),
            }.items():
                predictions = pooled_predictions(target_x, target_y, splitter, groups)
                feature_rows.append({
                    "audit": "linear grain-width control",
                    "target_or_term": target,
                    "model_or_statistic": name,
                    "protocol": protocol,
                    "value": r2_score(target_y, predictions),
                })

    campaign_c = all_records["Iteration"].str.startswith("C").astype(int)
    terms = {
        "HoldTime": all_records["HoldTime"],
        "dH_mix/HoldTime^2": all_records["dH_mix"] / all_records["HoldTime"] ** 2,
        "(6.93-d)/SD_grain": (6.93 - all_records["GrainSize"]) / all_records["SD_GS"],
    }
    for term, values in terms.items():
        feature_rows.append({
            "audit": "campaign separation",
            "target_or_term": term,
            "model_or_statistic": "direction-invariant ROC AUC",
            "protocol": "B versus C",
            "value": direction_invariant_auc(campaign_c, values),
        })

    feature_result = pd.DataFrame(feature_rows)
    feature_result.to_csv(OUT_FEATURE, index=False)
    print(feature_result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {OUT_FEATURE}")


if __name__ == "__main__":
    main()
