# build_summary.py
#
# Collapses this repo's per-(feature set, target, protocol) result CSVs into one
# master summary table, schema-matched to the companion repo's
# scripts/04_family4_nonlinear_ml/armote_ladder_cv.py output
# (results/armote_ladder_cv.csv):
#
#   Target, FeatureSet, Model, n_feat, R2_5fold, RMSE_5fold,
#   LOO_R2, LOO_RMSE, LOBO_R2, LOBO_RMSE, BIC, HPO
#
# No new model runs and no stored .pkl models are touched -- everything here is
# read directly from the already-written summary CSVs under
# S{1-4}_{YS,HV}_Results_{5_Fold,LOBO,LOO}_CV/, which live inside the
# ARMOTE-CV-MPEA-Hall-Petch submodule alongside this script (paths resolved
# via __file__ so this works regardless of invocation cwd, e.g. `make family4`).
# The feature manifest is read from this repo's own data/derived/ (same file,
# byte-identical to the submodule's copy) via the shared _config.py convention
# every other script here uses. Output goes to this repo's own results/ dir.
#
# BIC: computed only for the linear-parametric models (LinearRegression, Ridge,
# Lasso, ElasticNet, PCA_OLS), from the POOLED LOO out-of-fold MSE already
# reported in the LOO CSV's "Avg Test MSE (original units)" column (this equals
# RSS/n because run_loo.py sets pool_oof_metrics=True -- see armote_cv.py:923-926).
# BIC = n * log(RSS/n) + k * log(n), matching the companion repo's bic_linear().
# Left blank for every non-parametric model (tree/kernel/boosting/NN/ensemble),
# same as the companion repo's own convention.
#
# Usage: python build_summary.py [--out results/armote_cv_summary.csv]

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _config import DATA_DIR, RESULTS_DIR

SUBMODULE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ARMOTE-CV-MPEA-Hall-Petch"
)

FEATURE_SETS = ["S1", "S2", "S3", "S4"]
TARGETS = ["YS", "HV"]

# Models with an empty Optuna search space (fixed settings, no tuning).
NO_TUNE_MODELS = {"LinearRegression", "MLP", "PCA_OLS", "Dummy", "Stacking"}

# BIC effective parameter count (k). Only defined for linear-parametric models.
BIC_K = {
    "LinearRegression": lambda n_feat: n_feat + 1,
    "Ridge": lambda n_feat: n_feat + 1,
    "Lasso": lambda n_feat: n_feat + 1,
    "ElasticNet": lambda n_feat: n_feat + 1,
    "PCA_OLS": lambda n_feat: 6 + 1,  # 6 PCA components + intercept
}


def cols_for_blocks(manifest_df, blocks):
    return manifest_df.loc[manifest_df["block"].isin(blocks), "column"].tolist()


def n_feat_for(feat_name, manifest):
    s1 = cols_for_blocks(manifest, ["S1_grain"])
    s2 = cols_for_blocks(manifest, ["S2_wen"])
    s3 = cols_for_blocks(manifest, ["S3_proc"])
    s4 = cols_for_blocks(manifest, ["S4_comp", "S4_sss"])
    n = {"S1": s1, "S2": s1 + s2, "S3": s1 + s2 + s3, "S4": s1 + s2 + s3 + s4}
    return len(n[feat_name])


def find_csv(feat_name, target_name, protocol_dir_glob):
    pattern = os.path.join(
        SUBMODULE_ROOT,
        f"{feat_name}_{target_name}_Results_{protocol_dir_glob}",
        "*_results_*_fold_CV.csv",
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  WARNING: multiple CSVs matched {pattern}, using {matches[0]}")
    return matches[0]


def load_protocol(feat_name, target_name, protocol_dir_glob):
    path = find_csv(feat_name, target_name, protocol_dir_glob)
    if path is None:
        print(
            f"  Skipping {protocol_dir_glob}: no results yet for {feat_name}/{target_name}"
        )
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[row["Model"]] = {
            "R2": row["Avg Test R2"],
            "MSE": row["Avg Test MSE (original units)"],
        }
    return out


def bic(model_name, n_feat, loo_row, n_samples):
    if model_name not in BIC_K or loo_row is None or n_samples is None:
        return np.nan
    k = BIC_K[model_name](n_feat)
    rss_over_n = loo_row["MSE"]
    if rss_over_n is None or (isinstance(rss_over_n, float) and np.isnan(rss_over_n)):
        return np.nan
    return round(n_samples * np.log(rss_over_n) + k * np.log(n_samples), 1)


def n_samples_for_loo(feat_name, target_name):
    path = find_csv(feat_name, target_name, "LOO_CV")
    if path is None:
        return None
    df = pd.read_csv(path)
    # LOO fold count == sample count; recover it from any row's per-fold array.
    for _, row in df.iterrows():
        folds = json.loads(row["All Fold Test R2"])
        return len(folds)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / "armote_cv_summary.csv")
    )
    args = parser.parse_args()

    manifest = pd.read_csv(DATA_DIR / "inputs_feature_manifest.csv")

    rows = []
    for target in TARGETS:
        for feat_name in FEATURE_SETS:
            print(f"\n{feat_name} | {target}")
            n_feat = n_feat_for(feat_name, manifest)

            fold5 = load_protocol(feat_name, target, "5_Fold_CV")
            loo = load_protocol(feat_name, target, "LOO_CV")
            lobo = load_protocol(feat_name, target, "LOBO_CV")
            n_samples = n_samples_for_loo(feat_name, target)

            models = sorted(set(fold5) | set(loo) | set(lobo))
            for model_name in models:
                r5 = fold5.get(model_name)
                rl = loo.get(model_name)
                rb = lobo.get(model_name)

                rows.append(
                    {
                        "Target": target,
                        "FeatureSet": feat_name,
                        "Model": model_name,
                        "n_feat": n_feat,
                        "R2_5fold": r5["R2"] if r5 else np.nan,
                        "RMSE_5fold": np.sqrt(r5["MSE"]) if r5 else np.nan,
                        "LOO_R2": rl["R2"] if rl else np.nan,
                        "LOO_RMSE": np.sqrt(rl["MSE"]) if rl else np.nan,
                        "LOBO_R2": rb["R2"] if rb else np.nan,
                        "LOBO_RMSE": np.sqrt(rb["MSE"]) if rb else np.nan,
                        "BIC": bic(model_name, n_feat, rl, n_samples),
                        "HPO": "none"
                        if model_name in NO_TUNE_MODELS
                        else "optuna-nested-50",
                    }
                )

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(result)} rows)")


if __name__ == "__main__":
    main()
