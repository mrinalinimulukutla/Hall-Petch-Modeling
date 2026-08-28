#!/usr/bin/env python3
"""
Exhaustive Model Search for HEA Yield Strength Prediction
==========================================================
12+ models × 4 feature sets, Optuna HPO, LOO + LOBO CV, stacking ensemble.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (LeaveOneOut, LeaveOneGroupOut, KFold,
                                      cross_val_score, RepeatedKFold)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.dummy import DummyRegressor
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import optuna
import shap
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'
ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
BATCH_COLORS = {'BBA': '#D55E00', 'BBB': '#0072B2', 'BBC': '#009E73',
                'CBA': '#CC79A7', 'CBB': '#E69F00', 'CBC': '#56B4E9'}

t0_global = time.time()

# ============================================================
# 1. LOAD AND ENGINEER FEATURES
# ============================================================
print("=" * 70)
print("EXHAUSTIVE MODEL SEARCH FOR HEA YIELD STRENGTH")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
if 'eps_Labusch.1' in df.columns:
    df = df.drop(columns=['eps_Labusch.1'])

# Clean infinities
df['Omega'] = df['Omega'].clip(upper=100)
df = df.replace([np.inf, -np.inf], np.nan)

# --- New physics-informed features ---
# Composition × d^(-1/2) interactions
for el in ELEMENTS:
    df[f'{el}_x_dinv'] = df[f'{el}_frac'] * df['d_inv_sqrt']

# Descriptor ratios
df['VEC_over_delta'] = df['VEC'] / (df['delta'] + 1e-8)
df['mu_x_delta'] = df['mu_bar'] * df['delta']
df['dH_over_Tm'] = df['dH_mix'] / df['Tm_bar']
df['Omega_x_delta'] = df['Omega'] * df['delta']

# SSS × d^(-1/2) interactions
df['VLC_x_dinv'] = df['sigma_y0_VLC'] * df['d_inv_sqrt']
df['Labusch_x_dinv'] = df['sigma_Labusch'] * df['d_inv_sqrt']
df['TC_x_dinv'] = df['sigma_TC'] * df['d_inv_sqrt']

# Processing × grain interactions
df['RecrystT_x_dinv'] = df['RecrystT'] * df['d_inv_sqrt']
df['CW_x_dinv'] = df['ColdWork'] * df['d_inv_sqrt']

# Quadratic terms for top features
df['dinv_sq'] = df['d_inv_sqrt'] ** 2
df['V_frac_sq'] = df['V_frac'] ** 2

# --- Define feature sets ---
FEAT_CORE = [f'{el}_frac' for el in ELEMENTS] + ['d_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime']

FEAT_PHYSICS = FEAT_CORE + [
    'delta', 'VEC', 'dH_mix', 'dS_mix', 'Omega', 'mu_bar', 'delta_chi',
    'Tm_bar', 'Phi_VLC', 'eps_Labusch', 'a_bar',
    'sigma_y0_VLC', 'sigma_Labusch', 'sigma_TC', 'delta_Yang',
]

FEAT_INTERACTIONS = FEAT_PHYSICS + [
    f'{el}_x_dinv' for el in ELEMENTS
] + [
    'VEC_over_delta', 'mu_x_delta', 'dH_over_Tm', 'Omega_x_delta',
    'VLC_x_dinv', 'Labusch_x_dinv', 'TC_x_dinv',
    'RecrystT_x_dinv', 'CW_x_dinv', 'dinv_sq', 'V_frac_sq',
]

# Filter to YS data
df_ys = df.dropna(subset=['YS']).copy()
y_all = df_ys['YS'].values
groups_all = df_ys['Iteration'].values
n = len(y_all)
print(f"\nSamples with YS: {n}")
print(f"Batches: {df_ys['Iteration'].value_counts().to_dict()}")

# Build COMPACT set via mutual information
X_int = df_ys[FEAT_INTERACTIONS].fillna(0).values
mi_scores = mutual_info_regression(X_int, y_all, random_state=42)
mi_ranking = pd.Series(mi_scores, index=FEAT_INTERACTIONS).sort_values(ascending=False)
FEAT_COMPACT = mi_ranking.head(18).index.tolist()

print(f"\nFeature sets: CORE={len(FEAT_CORE)}, PHYSICS={len(FEAT_PHYSICS)}, "
      f"INTERACTIONS={len(FEAT_INTERACTIONS)}, COMPACT={len(FEAT_COMPACT)}")
print(f"COMPACT (top 18 by MI): {FEAT_COMPACT}")


def get_Xy(feat_set):
    """Return clean X, y arrays for a feature set."""
    cols = [c for c in feat_set if c in df_ys.columns]
    data = df_ys[cols + ['YS']].fillna(0)
    return data[cols].values, data['YS'].values, cols


# ============================================================
# 2. EVALUATION FUNCTIONS
# ============================================================

def eval_loo(model_factory, X, y):
    """Leave-One-Out cross-validation."""
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds), np.sqrt(mean_squared_error(y, preds)), mean_absolute_error(y, preds), preds


def eval_lobo(model_factory, X, y, groups):
    """Leave-One-Batch-Out cross-validation."""
    preds = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds), np.sqrt(mean_squared_error(y, preds)), mean_absolute_error(y, preds), preds


def eval_5fold(model_factory, X, y, seed=42):
    """5-fold KFold CV (same seed as cv_comparison.csv for direct comparison)."""
    from sklearn.model_selection import KFold
    preds = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds), np.sqrt(mean_squared_error(y, preds)), mean_absolute_error(y, preds), preds


def run_optuna(model_class, space_fn, X, y, n_trials=50, needs_scaling=False, timeout=300):
    """Run Optuna HPO with 5-fold CV."""
    def objective(trial):
        params = space_fn(trial)
        if needs_scaling:
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model_class(**params))])
        else:
            pipe = model_class(**params)
        scores = cross_val_score(pipe, X, y,
                                  cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),
                                  scoring='neg_mean_squared_error')
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params, -study.best_value


# ============================================================
# 3. MODEL DEFINITIONS
# ============================================================

MODEL_CONFIGS = {}

# --- No-HPO models ---
MODEL_CONFIGS['Baseline (mean)'] = {
    'factory': lambda: DummyRegressor(strategy='mean'),
    'feat': 'CORE', 'scale': False, 'hpo': False,
}
MODEL_CONFIGS['OLS'] = {
    'factory': lambda: Pipeline([('s', StandardScaler()), ('m', LinearRegression())]),
    'feat': 'PHYSICS', 'scale': False, 'hpo': False,
}
# --- Physics-informed Hall-Petch model (M3) ---
# σ_y = σ₀₀ + Σα_i·x_i + k·d⁻¹/² (OLS on intercept + 7 elem fracs + d_inv_sqrt)
FEAT_M3 = [f'{el}_frac' for el in ELEMENTS] + ['d_inv_sqrt']
MODEL_CONFIGS['M3: σ₀(all elem)'] = {
    'factory': lambda: LinearRegression(),
    'feat': 'M3', 'scale': False, 'hpo': False,
}

MODEL_CONFIGS['GPR (Matern)'] = {
    'feat': 'COMPACT', 'scale': True, 'hpo': False,
    'factory': lambda: Pipeline([('s', StandardScaler()), ('m', GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(len(FEAT_COMPACT)), nu=2.5)
        + WhiteKernel(noise_level=1.0),
        n_restarts_optimizer=15, random_state=42, normalize_y=True, alpha=1e-6))]),
}

# --- Optuna-tuned models ---

def ridge_space(trial):
    return {'alpha': trial.suggest_float('alpha', 1e-3, 100, log=True)}

def enet_space(trial):
    return {'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 0.99),
            'max_iter': 10000}

def svr_space(trial):
    return {'C': trial.suggest_float('C', 0.1, 1000, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 10, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])}

def krr_space(trial):
    return {'alpha': trial.suggest_float('alpha', 1e-3, 100, log=True),
            'kernel': 'rbf',
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True)}

def rf_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 15),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42}

def xgb_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 50, 700),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'objective': 'reg:squarederror', 'random_state': 42, 'verbosity': 0}

def cat_space(trial):
    return {'iterations': trial.suggest_int('iterations', 100, 800),
            'depth': trial.suggest_int('depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42, 'verbose': 0}

def lgbm_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 50, 700),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'random_state': 42, 'verbosity': -1}

OPTUNA_MODELS = {
    'Ridge': {'cls': Ridge, 'space': ridge_space, 'feat': 'PHYSICS', 'scale': True},
    'ElasticNet': {'cls': ElasticNet, 'space': enet_space, 'feat': 'PHYSICS', 'scale': True},
    'SVR (RBF)': {'cls': SVR, 'space': svr_space, 'feat': 'COMPACT', 'scale': True},
    'KRR (RBF)': {'cls': KernelRidge, 'space': krr_space, 'feat': 'COMPACT', 'scale': True},
    'Random Forest': {'cls': RandomForestRegressor, 'space': rf_space, 'feat': 'INTERACTIONS', 'scale': False},
    'XGBoost': {'cls': xgb.XGBRegressor, 'space': xgb_space, 'feat': 'INTERACTIONS', 'scale': False},
    'CatBoost': {'cls': cb.CatBoostRegressor, 'space': cat_space, 'feat': 'INTERACTIONS', 'scale': False},
    'LightGBM': {'cls': lgb.LGBMRegressor, 'space': lgbm_space, 'feat': 'INTERACTIONS', 'scale': False},
}

FEAT_MAP = {
    'CORE': FEAT_CORE, 'PHYSICS': FEAT_PHYSICS,
    'INTERACTIONS': FEAT_INTERACTIONS, 'COMPACT': FEAT_COMPACT,
    'M3': FEAT_M3,
}

# ============================================================
# 4. MAIN EVALUATION LOOP
# ============================================================
print("\n" + "=" * 70)
print("RUNNING MODEL EVALUATIONS")
print("=" * 70)

results = []
all_loo_preds = {}

# --- No-HPO models ---
for name, cfg in MODEL_CONFIGS.items():
    t0 = time.time()
    feat_key = cfg['feat']
    X, y, cols = get_Xy(FEAT_MAP[feat_key])
    groups = groups_all

    print(f"\n  [{name}] features={feat_key} ({X.shape[1]}), n={len(y)}...")

    r2_5f, rmse_5f, mae_5f, _ = eval_5fold(cfg['factory'], X, y)
    r2_loo, rmse_loo, mae_loo, preds_loo = eval_loo(cfg['factory'], X, y)
    r2_lobo, rmse_lobo, mae_lobo, preds_lobo = eval_lobo(cfg['factory'], X, y, groups)

    dt = time.time() - t0
    print(f"    5f R²={r2_5f:.4f} | LOO R²={r2_loo:.4f}, RMSE={rmse_loo:.1f} | LOBO R²={r2_lobo:.4f} | {dt:.1f}s")

    results.append({'Model': name, 'Features': feat_key, 'n_feat': X.shape[1],
                     'R2_5fold': r2_5f, 'RMSE_5fold': rmse_5f,
                     'LOO_R2': r2_loo, 'LOO_RMSE': rmse_loo, 'LOO_MAE': mae_loo,
                     'LOBO_R2': r2_lobo, 'LOBO_RMSE': rmse_lobo,
                     'HPO': 'None', 'Time_s': dt})
    all_loo_preds[name] = preds_loo

# --- TabPFN ---
# Disabled on macOS arm64. tabpfn auto-downloads HF transformer weights at
# fit time, and the download path segfaults without a HF_TOKEN. A native
# segfault bypasses Python's try/except and kills the whole process, so
# we can't merely wrap it. Original panel numbers show TabPFN LOO ~0.58
# which is below the linear/tree baselines anyway; skipping is OK.
print(f"\n  [TabPFN] skipped on macOS arm64 (HF auto-download segfault)")

# --- Optuna-tuned models ---
for name, cfg in OPTUNA_MODELS.items():
    t0 = time.time()
    feat_key = cfg['feat']
    X, y, cols = get_Xy(FEAT_MAP[feat_key])
    groups = groups_all
    cls = cfg['cls']
    space_fn = cfg['space']
    needs_scale = cfg['scale']

    print(f"\n  [{name}] features={feat_key} ({X.shape[1]}), Optuna 50 trials...")

    try:
        # Stage 1: Global HPO
        best_params, best_mse = run_optuna(cls, space_fn, X, y, n_trials=50,
                                            needs_scaling=needs_scale, timeout=300)
        print(f"    Best inner MSE={best_mse:.1f}, params={best_params}")

        # Factory for evaluation
        if needs_scale:
            factory = lambda bp=best_params: Pipeline([('s', StandardScaler()),
                                                        ('m', cls(**bp))])
        else:
            factory = lambda bp=best_params: cls(**bp)

        # 5-fold with fixed best params
        r2_5f, rmse_5f, mae_5f, _ = eval_5fold(factory, X, y)
        # LOO with fixed best params
        r2_loo, rmse_loo, mae_loo, preds_loo = eval_loo(factory, X, y)

        # LOBO with fixed best params
        r2_lobo, rmse_lobo, mae_lobo, preds_lobo = eval_lobo(factory, X, y, groups)

        dt = time.time() - t0
        print(f"    5f R²={r2_5f:.4f} | LOO R²={r2_loo:.4f}, RMSE={rmse_loo:.1f} | LOBO R²={r2_lobo:.4f} | {dt:.1f}s")

        results.append({'Model': name, 'Features': feat_key, 'n_feat': X.shape[1],
                         'R2_5fold': r2_5f, 'RMSE_5fold': rmse_5f,
                         'LOO_R2': r2_loo, 'LOO_RMSE': rmse_loo, 'LOO_MAE': mae_loo,
                         'LOBO_R2': r2_lobo, 'LOBO_RMSE': rmse_lobo,
                         'HPO': f'Optuna-50', 'Time_s': dt})
        all_loo_preds[name] = preds_loo

    except Exception as e:
        dt = time.time() - t0
        print(f"    FAILED: {e}")
        results.append({'Model': name, 'Features': feat_key, 'n_feat': X.shape[1],
                         'R2_5fold': np.nan, 'RMSE_5fold': np.nan,
                         'LOO_R2': np.nan, 'LOO_RMSE': np.nan, 'LOO_MAE': np.nan,
                         'LOBO_R2': np.nan, 'LOBO_RMSE': np.nan,
                         'HPO': 'FAILED', 'Time_s': dt})

# --- Also try XGBoost and CatBoost on COMPACT features ---
for name_suffix, cfg_name in [('XGBoost-compact', 'XGBoost'), ('CatBoost-compact', 'CatBoost')]:
    t0 = time.time()
    cfg = OPTUNA_MODELS[cfg_name]
    X, y, cols = get_Xy(FEAT_COMPACT)
    groups = groups_all
    cls = cfg['cls']
    space_fn = cfg['space']

    print(f"\n  [{name_suffix}] features=COMPACT ({X.shape[1]}), Optuna 50 trials...")

    try:
        best_params, best_mse = run_optuna(cls, space_fn, X, y, n_trials=50,
                                            needs_scaling=False, timeout=300)
        factory = lambda bp=best_params: cls(**bp)
        r2_5f, rmse_5f, mae_5f, _ = eval_5fold(factory, X, y)
        r2_loo, rmse_loo, mae_loo, preds_loo = eval_loo(factory, X, y)
        r2_lobo, rmse_lobo, mae_lobo, preds_lobo = eval_lobo(factory, X, y, groups)

        dt = time.time() - t0
        print(f"    5f R²={r2_5f:.4f} | LOO R²={r2_loo:.4f}, RMSE={rmse_loo:.1f} | LOBO R²={r2_lobo:.4f} | {dt:.1f}s")

        results.append({'Model': name_suffix, 'Features': 'COMPACT', 'n_feat': X.shape[1],
                         'R2_5fold': r2_5f, 'RMSE_5fold': rmse_5f,
                         'LOO_R2': r2_loo, 'LOO_RMSE': rmse_loo, 'LOO_MAE': mae_loo,
                         'LOBO_R2': r2_lobo, 'LOBO_RMSE': rmse_lobo,
                         'HPO': 'Optuna-50', 'Time_s': dt})
        all_loo_preds[name_suffix] = preds_loo
    except Exception as e:
        dt = time.time() - t0
        print(f"    FAILED: {e}")

# ============================================================
# 5. STACKING ENSEMBLE
# ============================================================
print("\n" + "=" * 70)
print("STACKING ENSEMBLE")
print("=" * 70)

# Select top 5 diverse base models
res_df = pd.DataFrame(results).dropna(subset=['LOO_R2']).sort_values('LOO_R2', ascending=False)
print("\nAll models ranked by LOO R²:")
print(res_df[['Model', 'Features', 'n_feat', 'LOO_R2', 'LOO_RMSE', 'LOBO_R2']].to_string(index=False))

# Pick diverse top models for stacking
model_families = {
    'tree': ['XGBoost', 'XGBoost-compact', 'Random Forest', 'LightGBM'],
    'boosting2': ['CatBoost', 'CatBoost-compact'],
    'kernel': ['GPR (Matern)', 'SVR (RBF)', 'KRR (RBF)'],
    'linear': ['Ridge', 'ElasticNet', 'OLS'],
    'physics': ['M3: σ₀(all elem)'],
    'foundation': ['TabPFN'],
}

# Select best from each family
stack_models = []
for family_name, candidates in model_families.items():
    avail = [c for c in candidates if c in all_loo_preds]
    if avail:
        best_in_family = max(avail, key=lambda m: res_df[res_df['Model'] == m]['LOO_R2'].values[0]
                              if len(res_df[res_df['Model'] == m]) > 0 else -np.inf)
        stack_models.append(best_in_family)
        print(f"  {family_name}: {best_in_family}")

if len(stack_models) >= 3:
    # Build meta-features from LOO predictions
    meta_X = np.column_stack([all_loo_preds[m] for m in stack_models])
    print(f"\nStacking {len(stack_models)} models: {stack_models}")
    print(f"  Meta-feature matrix: {meta_X.shape}")

    # LOO on Ridge meta-learner (doubly out-of-sample)
    stack_preds = np.zeros(n)
    for tr, te in LeaveOneOut().split(meta_X):
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge.fit(meta_X[tr], y_all[tr])
        stack_preds[te] = ridge.predict(meta_X[te])

    r2_stack = r2_score(y_all, stack_preds)
    rmse_stack = np.sqrt(mean_squared_error(y_all, stack_preds))
    mae_stack = mean_absolute_error(y_all, stack_preds)

    # LOBO for stacking
    stack_preds_lobo = np.zeros(n)
    for tr, te in LeaveOneGroupOut().split(meta_X, y_all, groups_all):
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge.fit(meta_X[tr], y_all[tr])
        stack_preds_lobo[te] = ridge.predict(meta_X[te])
    r2_stack_lobo = r2_score(y_all, stack_preds_lobo)
    rmse_stack_lobo = np.sqrt(mean_squared_error(y_all, stack_preds_lobo))

    print(f"  Stack LOO:  R²={r2_stack:.4f}, RMSE={rmse_stack:.1f}")
    print(f"  Stack LOBO: R²={r2_stack_lobo:.4f}, RMSE={rmse_stack_lobo:.1f}")

    results.append({'Model': 'Stacking (Ridge)', 'Features': 'meta', 'n_feat': len(stack_models),
                     'LOO_R2': r2_stack, 'LOO_RMSE': rmse_stack, 'LOO_MAE': mae_stack,
                     'LOBO_R2': r2_stack_lobo, 'LOBO_RMSE': rmse_stack_lobo,
                     'HPO': 'RidgeCV', 'Time_s': 0})
    all_loo_preds['Stacking (Ridge)'] = stack_preds

    # Simple average ensemble
    avg_preds = np.mean(meta_X, axis=1)
    r2_avg = r2_score(y_all, avg_preds)
    rmse_avg = np.sqrt(mean_squared_error(y_all, avg_preds))
    mae_avg = mean_absolute_error(y_all, avg_preds)
    print(f"  Average ensemble LOO: R²={r2_avg:.4f}, RMSE={rmse_avg:.1f}")

    results.append({'Model': 'Average Ensemble', 'Features': 'meta', 'n_feat': len(stack_models),
                     'LOO_R2': r2_avg, 'LOO_RMSE': rmse_avg, 'LOO_MAE': mae_avg,
                     'LOBO_R2': np.nan, 'LOBO_RMSE': np.nan,
                     'HPO': 'None', 'Time_s': 0})

# ============================================================
# 6. FINAL RESULTS TABLE
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

res_df = pd.DataFrame(results).sort_values('LOO_R2', ascending=False).reset_index(drop=True)
print(f"\n{'Rank':>4s}  {'Model':<22s} {'Feat':<14s} {'nF':>3s} {'LOO R²':>8s} {'LOO RMSE':>9s} "
      f"{'LOO MAE':>8s} {'LOBO R²':>8s} {'HPO':<10s} {'Time':>6s}")
print("  " + "-" * 105)
for i, row in res_df.iterrows():
    loo_r2 = f"{row['LOO_R2']:.4f}" if not np.isnan(row['LOO_R2']) else "   N/A"
    lobo_r2 = f"{row['LOBO_R2']:.4f}" if not np.isnan(row.get('LOBO_R2', np.nan)) else "   N/A"
    print(f"  {i+1:>2d}.  {row['Model']:<22s} {row['Features']:<14s} {row['n_feat']:>3.0f} "
          f"{loo_r2:>8s} {row['LOO_RMSE']:>9.1f} {row['LOO_MAE']:>8.1f} "
          f"{lobo_r2:>8s} {str(row['HPO']):<10s} {row['Time_s']:>5.0f}s")

res_df.to_csv(f'{RESULTS_DIR}/model_search_results.csv', index=False)
print(f"\nResults saved to model_search_results.csv")

# ============================================================
# 7. VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# --- 20: Model comparison bar chart ---
fig, ax = plt.subplots(figsize=(12, 8))
plot_df = res_df.dropna(subset=['LOO_R2']).copy()
colors = []
for m in plot_df['Model']:
    if 'Stack' in m or 'Average' in m:
        colors.append('#D32F2F')
    elif 'TabPFN' in m:
        colors.append('#7B1FA2')
    elif any(t in m for t in ['XGB', 'Cat', 'Light', 'Random', 'Gradient']):
        colors.append('#F57C00')
    elif any(t in m for t in ['GPR', 'SVR', 'KRR']):
        colors.append('#388E3C')
    else:
        colors.append('#1976D2')

ax.barh(range(len(plot_df)), plot_df['LOO_R2'].values, color=colors[::-1])
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df['Model'].values[::-1], fontsize=10)
ax.set_xlabel('LOO R²', fontsize=13)
ax.set_title('Model Comparison: LOO R² for Yield Strength', fontsize=15)
ax.axvline(x=0.644, color='gray', linestyle='--', alpha=0.7, label='Previous best (0.644)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/20_model_comparison_bar.png', dpi=150)
plt.close()
print("  Saved 20_model_comparison_bar.png")

# --- 21: Parity grid ---
top_models = res_df.dropna(subset=['LOO_R2']).head(12)['Model'].tolist()
top_models = [m for m in top_models if m in all_loo_preds]
n_plots = min(len(top_models), 12)
ncols = 4
nrows = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
if nrows == 1:
    axes = axes.reshape(1, -1)

for idx in range(nrows * ncols):
    ax = axes[idx // ncols, idx % ncols]
    if idx < n_plots:
        m_name = top_models[idx]
        preds = all_loo_preds[m_name]
        r2 = r2_score(y_all, preds)
        rmse = np.sqrt(mean_squared_error(y_all, preds))

        for batch, color in BATCH_COLORS.items():
            mask = groups_all == batch
            if mask.sum() > 0:
                ax.scatter(y_all[mask], preds[mask], c=color, s=25, alpha=0.8,
                           edgecolors='k', linewidth=0.3, label=batch)

        lims = [min(y_all.min(), preds.min()) * 0.9, max(y_all.max(), preds.max()) * 1.1]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_title(f'{m_name}\nR²={r2:.3f}, RMSE={rmse:.0f}', fontsize=10)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')
    else:
        ax.axis('off')

fig.supxlabel('Experimental YS (MPa)', fontsize=13)
fig.supylabel('Predicted YS (MPa)', fontsize=13)
plt.suptitle('LOO Parity Plots — Top Models', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/21_parity_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 21_parity_grid.png")

# --- 22: Best model parity ---
best_model = res_df.iloc[0]['Model']
if best_model in all_loo_preds:
    fig, ax = plt.subplots(figsize=(8, 8))
    preds = all_loo_preds[best_model]
    r2 = r2_score(y_all, preds)
    rmse = np.sqrt(mean_squared_error(y_all, preds))
    residuals = preds - y_all

    for batch, color in BATCH_COLORS.items():
        mask = groups_all == batch
        if mask.sum() > 0:
            ax.scatter(y_all[mask], preds[mask], c=color, s=60, alpha=0.8,
                       edgecolors='k', linewidth=0.5, label=batch)

    lims = [min(y_all.min(), preds.min()) * 0.9, max(y_all.max(), preds.max()) * 1.1]
    ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.set_xlabel('Experimental YS (MPa)', fontsize=13)
    ax.set_ylabel('Predicted YS (MPa)', fontsize=13)
    ax.set_title(f'Best Model: {best_model}\nLOO R²={r2:.4f}, RMSE={rmse:.1f} MPa', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/22_best_parity.png', dpi=150)
    plt.close()
    print("  Saved 22_best_parity.png")

# --- 23: LOO vs LOBO scatter ---
fig, ax = plt.subplots(figsize=(8, 8))
plot_both = res_df.dropna(subset=['LOO_R2', 'LOBO_R2'])
for _, row in plot_both.iterrows():
    ax.scatter(row['LOO_R2'], row['LOBO_R2'], s=80, zorder=3, edgecolors='k', linewidth=0.5)
    ax.annotate(row['Model'], (row['LOO_R2'], row['LOBO_R2']),
                textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)
lims = [min(plot_both[['LOO_R2', 'LOBO_R2']].min().min() - 0.1, -0.1),
        max(plot_both[['LOO_R2', 'LOBO_R2']].max().max() + 0.05, 0.8)]
ax.plot(lims, lims, 'k--', alpha=0.5)
ax.set_xlabel('LOO R²', fontsize=13)
ax.set_ylabel('LOBO R² (Leave-One-Batch-Out)', fontsize=13)
ax.set_title('Generalization: LOO vs LOBO Cross-Validation', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/23_loo_vs_lobo.png', dpi=150)
plt.close()
print("  Saved 23_loo_vs_lobo.png")

# --- 24: SHAP for best tree model ---
best_tree = None
for m in res_df['Model']:
    if m in ['XGBoost', 'CatBoost', 'LightGBM', 'XGBoost-compact', 'CatBoost-compact']:
        best_tree = m
        break

if best_tree:
    print(f"\n  Computing SHAP for {best_tree}...")
    cfg = OPTUNA_MODELS.get(best_tree.replace('-compact', ''), OPTUNA_MODELS.get(best_tree))
    feat_key = 'COMPACT' if 'compact' in best_tree else cfg['feat']
    X_shap, y_shap, cols_shap = get_Xy(FEAT_MAP[feat_key])
    best_params_shap, _ = run_optuna(cfg['cls'], cfg['space'], X_shap, y_shap,
                                      n_trials=30, needs_scaling=False, timeout=120)
    model_shap = cfg['cls'](**best_params_shap)
    model_shap.fit(X_shap, y_shap)

    explainer = shap.TreeExplainer(model_shap)
    shap_vals = explainer.shap_values(X_shap)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_shap, feature_names=cols_shap, show=False, max_display=20)
    plt.title(f'SHAP Summary — {best_tree}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/24_best_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 24_best_shap.png")

# ============================================================
# DONE
# ============================================================
total_time = time.time() - t0_global
print("\n" + "=" * 70)
print(f"EXHAUSTIVE MODEL SEARCH COMPLETE ({total_time:.0f}s)")
print("=" * 70)
print(f"\nBest model: {res_df.iloc[0]['Model']} — LOO R²={res_df.iloc[0]['LOO_R2']:.4f}")
print(f"Previous best: XGBoost LOO R²=0.644")
improvement = res_df.iloc[0]['LOO_R2'] - 0.644
print(f"Improvement: {improvement:+.4f} ({improvement/0.644*100:+.1f}%)")
