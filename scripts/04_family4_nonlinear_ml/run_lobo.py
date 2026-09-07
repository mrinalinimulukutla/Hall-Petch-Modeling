# run_lobo.py
#
# Runs ARMOTE-CV workflow with Leave-One-Batch-Out (LOBO) CV.
#
# Outer CV : LeaveOneGroupOut() grouped by the "Iteration" column
#            (6 batches: BBA/BBB/BBC/CBA/CBB/CBC). One whole batch held out per fold.
# Inner CV : KFold(n_splits=inner_cv) — used by Optuna only.
# Metrics  : R²/MSE/MAPE computed on pooled OOF predictions across all 6 folds.
#            Measures cross-batch extrapolation ability.
#
# Usage examples:
#   python run_lobo.py                              # all feature sets, targets, models
#   python run_lobo.py -f S1 S2 -t YS              # S1+S2, YS only
#   python run_lobo.py -f S3 -t HV -m SVR XGBoost  # S3, HV, two models only
#   python run_lobo.py --list                       # print valid choices and exit

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

try:
    from armote_cv import run_workflow
except ImportError:
    print("=" * 80)
    print("ERROR: 'armote_cv.py' not found in the current directory.")
    print("=" * 80)
    raise

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Data loading ───────────────────────────────────────────────────────────────
# Uses this repo's own data/derived/ (same file, byte-identical to the ARMOTE-CV
# submodule's copy) via the shared _config.py convention every other script here uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _config import DATA_DIR

df = pd.read_csv(DATA_DIR / "inputs.csv")
manifest = pd.read_csv(DATA_DIR / "inputs_feature_manifest.csv")


# ── Feature sets (built from manifest, cumulative) ────────────────────────────
def cols_for_blocks(manifest_df, blocks):
    return manifest_df.loc[manifest_df["block"].isin(blocks), "column"].tolist()


s1 = cols_for_blocks(manifest, ["S1_grain"])
s2 = cols_for_blocks(manifest, ["S2_wen"])
s3 = cols_for_blocks(manifest, ["S3_proc"])
s4 = cols_for_blocks(manifest, ["S4_comp", "S4_sss"])

ALL_FEATURE_SETS = {
    "S1": s1,
    "S2": s1 + s2,
    "S3": s1 + s2 + s3,
    "S4": s1 + s2 + s3 + s4,
}

ALL_TARGETS = ["YS", "HV"]

# ── Models ─────────────────────────────────────────────────────────────────────
ALL_MODELS = {
    "LinearRegression": LinearRegression(),
    "BayesianRidge": BayesianRidge(),
    "SVR": SVR(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=1),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=1),
    "GPR": GaussianProcessRegressor(random_state=42, n_restarts_optimizer=9),
    "NNR": None,
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(max_iter=10000, random_state=42),
    "ElasticNet": ElasticNet(max_iter=10000, random_state=42),
    "KernelRidge": KernelRidge(kernel="rbf"),
    "ExtraTrees": ExtraTreesRegressor(random_state=42, n_jobs=1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42, n_jobs=1, verbose=-1),
    "CatBoost": CatBoostRegressor(
        random_state=42,
        verbose=0,
        allow_writing_files=False,
        thread_count=1,
        task_type="CPU",
    ),
    # Fixed settings, matching Hall-Petch-Modeling/scripts/04_family4_nonlinear_ml/fair_comparison.py
    "MLP": MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=42
    ),
    "PCA_OLS": Pipeline(
        [("pca", PCA(n_components=6, random_state=42)), ("ols", LinearRegression())]
    ),
    "Dummy": DummyRegressor(strategy="mean"),
    # All parallelism lives at the Optuna trial level (armote_cv.py's outer
    # study.optimize(n_jobs=-1)); every model/CV layer below it is forced to
    # n_jobs=1 / thread_count=1 to avoid oversubscribing cores (was Optuna
    # trials x inner-CV folds x per-model threads, all at -1).
    "Stacking": StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(random_state=42, n_jobs=1)),
            ("xgb", XGBRegressor(random_state=42, n_jobs=1)),
            ("lgbm", LGBMRegressor(random_state=42, n_jobs=1, verbose=-1)),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20)),
        n_jobs=1,
    ),
}

# ── GPR kernel map ─────────────────────────────────────────────────────────────
gpr_kernel_map = {
    "RBF_default": RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
    "Matern_nu_0.5": Matern(length_scale=1.0, nu=0.5, length_scale_bounds=(1e-2, 1e2)),
    "Matern_nu_1.5": Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)),
    "Matern_nu_2.5": Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)),
    "RationalQuadratic": RationalQuadratic(
        length_scale=1.0,
        alpha=0.1,
        length_scale_bounds=(1e-2, 1e2),
        alpha_bounds=(1e-2, 1e2),
    ),
}

# ── Hyperparameter search spaces ───────────────────────────────────────────────
param_spaces = {
    "LinearRegression": {},
    "BayesianRidge": {
        "max_iter": ("int", 100, 500),
        "alpha_1": ("float", 1e-7, 1e-5, "log"),
        "alpha_2": ("float", 1e-7, 1e-5, "log"),
        "lambda_1": ("float", 1e-7, 1e-5, "log"),
        "lambda_2": ("float", 1e-7, 1e-5, "log"),
    },
    "SVR": {
        "C": ("float", 0.1, 1e4, "log"),
        "gamma": ("float", 1e-4, 1.0, "log"),
        "epsilon": ("float", 1e-3, 0.5, "log"),
    },
    "DecisionTree": {
        "max_depth": ("int", 5, 50),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 20),
    },
    "RandomForest": {
        "n_estimators": ("int", 50, 200),
        "max_depth": ("int", 5, 50),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "XGBoost": {
        "n_estimators": ("int", 50, 200),
        "learning_rate": ("float", 0.01, 0.5, "log"),
        "max_depth": ("int", 3, 10),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "GPR": {
        "kernel": ("categorical", list(gpr_kernel_map.keys())),
        "alpha": ("float", 1e-10, 1e-1, "log"),
    },
    "NNR": {
        "hidden_layers": ("int", 1, 4),
        "units": ("int", 32, 128),
        "activation": ("categorical", ["relu", "tanh", "selu"]),
        "learning_rate": ("float", 1e-5, 1e-2, "log"),
    },
    "Ridge": {
        "alpha": ("float", 1e-3, 1e3, "log"),
    },
    "Lasso": {
        "alpha": ("float", 1e-4, 10.0, "log"),
    },
    "ElasticNet": {
        "alpha": ("float", 1e-4, 10.0, "log"),
        "l1_ratio": ("float", 0.05, 0.95),
    },
    "KernelRidge": {
        "alpha": ("float", 1e-3, 10.0, "log"),
        "gamma": ("float", 1e-4, 1.0, "log"),
    },
    "ExtraTrees": {
        "n_estimators": ("int", 50, 400),
        "max_depth": ("int", 5, 50),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "GradientBoosting": {
        "n_estimators": ("int", 50, 400),
        "learning_rate": ("float", 0.01, 0.3, "log"),
        "max_depth": ("int", 2, 8),
        "subsample": ("float", 0.6, 1.0),
    },
    "LightGBM": {
        "n_estimators": ("int", 50, 700),
        "max_depth": ("int", 2, 6),
        "learning_rate": ("float", 0.005, 0.2, "log"),
        "num_leaves": ("int", 7, 63),
        "min_child_samples": ("int", 5, 30),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
    },
    "CatBoost": {
        "iterations": ("int", 50, 500),
        "depth": ("int", 3, 10),
        "learning_rate": ("float", 0.01, 0.3, "log"),
        "l2_leaf_reg": ("float", 1.0, 10.0, "log"),
    },
    # Fixed (no Optuna search), matching the companion repo's zero-tuning design
    "MLP": {},
    "PCA_OLS": {},
    "Dummy": {},
    "Stacking": {},
}

# ── CLI argument parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run ARMOTE-CV LOBO workflow.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-f", "--feature-sets",
    nargs="+",
    choices=list(ALL_FEATURE_SETS.keys()),
    default=list(ALL_FEATURE_SETS.keys()),
    metavar="FEAT",
    help=f"Feature sets to run. Choices: {list(ALL_FEATURE_SETS.keys())} (default: all)",
)
parser.add_argument(
    "-t", "--targets",
    nargs="+",
    choices=ALL_TARGETS,
    default=ALL_TARGETS,
    metavar="TARGET",
    help=f"Targets to run. Choices: {ALL_TARGETS} (default: all)",
)
parser.add_argument(
    "-m", "--models",
    nargs="+",
    choices=list(ALL_MODELS.keys()),
    default=list(ALL_MODELS.keys()),
    metavar="MODEL",
    help=f"Models to run. Choices: {list(ALL_MODELS.keys())} (default: all)",
)
parser.add_argument(
    "--n-trials",
    type=int,
    default=50,
    help="Optuna trials per inner fold (default: 50)",
)
parser.add_argument(
    "--inner-cv",
    type=int,
    default=5,
    help="KFold folds for Optuna inner CV (default: 5)",
)
parser.add_argument(
    "--list",
    action="store_true",
    help="Print valid choices for -f / -t / -m and exit.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing output CSV files instead of raising an error.",
)
parser.add_argument(
    "--append",
    action="store_true",
    help="Append results for missing models to an existing output CSV.",
)

args = parser.parse_args()

if args.list:
    print("Feature sets :", list(ALL_FEATURE_SETS.keys()))
    print("Targets      :", ALL_TARGETS)
    print("Models       :", list(ALL_MODELS.keys()))
    sys.exit(0)

# Build filtered selections
feature_sets = {k: ALL_FEATURE_SETS[k] for k in args.feature_sets}
targets = args.targets
models_to_run = {k: ALL_MODELS[k] for k in args.models}

# ── Main loop ──────────────────────────────────────────────────────────────────
total_runs = len(feature_sets) * len(targets)
print("=" * 80)
print("STARTING ARMOTE-CV LOBO WORKFLOWS")
print(f"Feature sets : {list(feature_sets.keys())}")
print(f"Targets      : {targets}")
print(f"Models       : {list(models_to_run.keys())}")
print(f"Inner CV     : {args.inner_cv}-fold KFold (Optuna only)")
print(f"Optuna trials: {args.n_trials} per inner fold  ({args.inner_cv * args.n_trials} total per model per outer fold)")
print(f"Total runs   : {total_runs}")
print("=" * 80)

for feat_name, feat_cols in feature_sets.items():
    for target_name in targets:
        print(f"\n{'=' * 30} {feat_name} | Target: {target_name} {'=' * 30}")

        subset = df[feat_cols + [target_name, "Iteration"]].dropna()
        n_dropped = len(df) - len(subset)
        print(
            f"Samples: {len(subset)} / {len(df)} ({n_dropped} NaN rows dropped for {target_name})"
        )
        print(f"Features ({len(feat_cols)}): {feat_cols}")

        X = subset[feat_cols]
        y = subset[target_name].values
        groups = subset["Iteration"].values

        batches = sorted(set(groups))
        print(f"LOBO batches ({len(batches)}): {batches}")

        # PCA_OLS(6) needs >= 6 input features (matches Hall-Petch-Modeling's own guard)
        models_this_run = dict(models_to_run)
        if "PCA_OLS" in models_this_run and len(feat_cols) < 6:
            print(f"Skipping PCA_OLS: {feat_name} has {len(feat_cols)} features (< 6 required)")
            del models_this_run["PCA_OLS"]

        output_folder = f"{feat_name}_{target_name}_Results_LOBO_CV"

        run_workflow(
            X,
            y,
            models_this_run,
            param_spaces,
            output_name=target_name,
            output_folder_name=output_folder,
            gpr_kernel_map=gpr_kernel_map,
            nn_epochs=100,
            nn_batch_size=8,
            colors=["#EE6677", "#228833", "#4477AA", "#CCBB44", "#66CCEE"],
            refit_scaler_per_fold=True,
            n_trials=args.n_trials,
            cv_random_state=42,
            nn_model_names=("NNR",),
            gpr_model_names=("GPR",),
            splitter=LeaveOneGroupOut(),
            inner_cv=args.inner_cv,
            pool_oof_metrics=True,
            groups=groups,
            overwrite=args.overwrite,
            append=args.append,
        )

        print(f"{'=' * 30} Completed: {feat_name} | {target_name} {'=' * 30}\n")

print("\n" + "=" * 80)
print("ALL LOBO WORKFLOWS COMPLETE.")
print("=" * 80)
