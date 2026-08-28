#!/usr/bin/env python3
"""
Grain-Size Scaling & Information-Criteria Model Comparison
===========================================================
1. Tests 8 alternative grain-size scaling laws against Hall-Petch d^(-1/2)
2. Compares via AIC, AICc, BIC, and LOO R²
3. Adds best alternative features and re-runs exhaustive model search
4. Reports AIC/BIC for all parametric models, LOO for all models
"""

import time, warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from sklearn.linear_model import (LinearRegression, Ridge, ElasticNet,
                                   RidgeCV, LassoCV)
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (LeaveOneOut, LeaveOneGroupOut,
                                      cross_val_score, RepeatedKFold)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.dummy import DummyRegressor
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import optuna
import shap

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
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("GRAIN-SIZE SCALING & INFORMATION-CRITERIA ANALYSIS")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
if 'eps_Labusch.1' in df.columns:
    df = df.drop(columns=['eps_Labusch.1'])
df['Omega'] = df['Omega'].clip(upper=100)
df = df.replace([np.inf, -np.inf], np.nan)

df_ys = df.dropna(subset=['YS']).copy()
y = df_ys['YS'].values
n = len(y)
d = df_ys['GrainSize'].values  # grain size in µm
groups = df_ys['Iteration'].values
print(f"\nSamples with YS: {n}")
print(f"Grain size range: {d.min():.1f} – {d.max():.1f} µm")

# ============================================================
# 2. INFORMATION CRITERIA FUNCTIONS
# ============================================================
def compute_ic(y_true, y_pred, k, n):
    """Compute AIC, AICc, BIC for a model with k parameters fitted to n points.

    Uses Gaussian likelihood: L = -n/2·ln(2π) - n/2·ln(RSS/n) - n/2
    => -2·ln(L) = n·ln(RSS/n) + n·ln(2π) + n
    AIC  = n·ln(RSS/n) + 2·k  (dropping constants common to all models)
    BIC  = n·ln(RSS/n) + k·ln(n)
    AICc = AIC + 2·k·(k+1) / (n-k-1)
    """
    rss = np.sum((y_true - y_pred) ** 2)
    if rss <= 0:
        rss = 1e-15
    log_term = n * np.log(rss / n)
    aic = log_term + 2 * k
    bic = log_term + k * np.log(n)
    if n - k - 1 > 0:
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = np.inf
    return {'AIC': aic, 'AICc': aicc, 'BIC': bic, 'RSS': rss}


def loo_r2(X, y, model_factory):
    """Leave-one-out cross-validated R²."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return r2, rmse, preds


# ============================================================
# 3. PHASE 1: GRAIN-SIZE SCALING COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: GRAIN-SIZE SCALING LAW COMPARISON")
print("=" * 70)
print("Comparing: σ_y = a + b·f(d) for various f(d)")

# Create all grain-size features
gs_features = {}
gs_features['d^(-1/2)  [Hall-Petch]'] = d ** (-0.5)
gs_features['d^(-1)    [Dunstan-Bushby]'] = d ** (-1.0)
gs_features['d^(-1/3)  [Baldwin]'] = d ** (-1.0 / 3.0)
gs_features['d^(-2/3)'] = d ** (-2.0 / 3.0)
gs_features['ln(d)/d   [Crit. thickness]'] = np.log(d) / d
gs_features['1/√d + 1/d [Composite]'] = None  # 2-feature model, handled separately
gs_features['ln(d)'] = np.log(d)
gs_features['d^(-1) + d^(-2) [Taylor exp.]'] = None  # 2-feature model

# Fit optimal exponent: σ = a + b·d^(-n_opt)
def neg_r2_for_exp(exp, d_arr=d, y_arr=y):
    f = d_arr ** (-exp)
    X_ = np.column_stack([np.ones(len(y_arr)), f])
    beta = np.linalg.lstsq(X_, y_arr, rcond=None)[0]
    pred = X_ @ beta
    return -r2_score(y_arr, pred)

result = minimize_scalar(neg_r2_for_exp, bounds=(0.01, 2.0), method='bounded')
n_opt = result.x
gs_features[f'd^(-{n_opt:.3f}) [Optimized]'] = d ** (-n_opt)

print(f"\n  Optimal exponent n = {n_opt:.3f} (fitted to maximize train R²)")

# Evaluate each scaling law
scaling_results = []

for name, feat in gs_features.items():
    if feat is None:
        continue  # multi-feature models handled below

    X_gs = feat.reshape(-1, 1)
    # Training fit
    reg = LinearRegression().fit(X_gs, y)
    y_pred_train = reg.predict(X_gs)
    k_params = 2  # intercept + slope
    ic = compute_ic(y, y_pred_train, k_params, n)

    # LOO R²
    r2_loo, rmse_loo, preds_loo = loo_r2(X_gs, y, LinearRegression)

    scaling_results.append({
        'Scaling': name,
        'k': k_params,
        'Train_R2': r2_score(y, y_pred_train),
        'LOO_R2': r2_loo,
        'LOO_RMSE': rmse_loo,
        'AIC': ic['AIC'],
        'AICc': ic['AICc'],
        'BIC': ic['BIC'],
        'intercept': reg.intercept_,
        'slope': reg.coef_[0],
    })

# Multi-feature models
# (a) Composite: d^(-1/2) + d^(-1)
X_comp = np.column_stack([d ** (-0.5), d ** (-1.0)])
reg_comp = LinearRegression().fit(X_comp, y)
y_pred_comp = reg_comp.predict(X_comp)
k_comp = 3
ic_comp = compute_ic(y, y_pred_comp, k_comp, n)
r2_comp, rmse_comp, _ = loo_r2(X_comp, y, LinearRegression)
scaling_results.append({
    'Scaling': '1/√d + 1/d [Composite]',
    'k': k_comp,
    'Train_R2': r2_score(y, y_pred_comp),
    'LOO_R2': r2_comp,
    'LOO_RMSE': rmse_comp,
    'AIC': ic_comp['AIC'],
    'AICc': ic_comp['AICc'],
    'BIC': ic_comp['BIC'],
    'intercept': reg_comp.intercept_,
    'slope': reg_comp.coef_[0],
})

# (b) d^(-1) + d^(-2)  (Taylor expansion / higher-order)
X_taylor = np.column_stack([d ** (-1.0), d ** (-2.0)])
reg_tay = LinearRegression().fit(X_taylor, y)
y_pred_tay = reg_tay.predict(X_taylor)
k_tay = 3
ic_tay = compute_ic(y, y_pred_tay, k_tay, n)
r2_tay, rmse_tay, _ = loo_r2(X_taylor, y, LinearRegression)
scaling_results.append({
    'Scaling': 'd^(-1) + d^(-2) [Taylor exp.]',
    'k': k_tay,
    'Train_R2': r2_score(y, y_pred_tay),
    'LOO_R2': r2_tay,
    'LOO_RMSE': rmse_tay,
    'AIC': ic_tay['AIC'],
    'AICc': ic_tay['AICc'],
    'BIC': ic_tay['BIC'],
    'intercept': reg_tay.intercept_,
    'slope': reg_tay.coef_[0],
})

# (c) Optimized exponent (3 params: intercept, k, n — already fit n above)
X_nopt = (d ** (-n_opt)).reshape(-1, 1)
reg_nopt = LinearRegression().fit(X_nopt, y)
y_pred_nopt = reg_nopt.predict(X_nopt)
k_nopt = 3  # extra param for the fitted exponent
ic_nopt = compute_ic(y, y_pred_nopt, k_nopt, n)

# Unbiased LOO for optimized exponent: re-optimize n inside each fold
# (avoids data leakage from fitting n_opt on all data then evaluating LOO)
loo_opt = LeaveOneOut()
preds_opt_unbiased = np.zeros(n)
for tr_idx, te_idx in loo_opt.split(d):
    d_tr, y_tr = d[tr_idx], y[tr_idx]
    d_te = d[te_idx]
    # Re-optimize exponent on training fold only
    res_fold = minimize_scalar(neg_r2_for_exp, bounds=(0.01, 2.0),
                               method='bounded',
                               args=(d_tr, y_tr))
    n_fold = res_fold.x
    # Fit linear model with fold-specific exponent on training data
    X_tr_fold = (d_tr ** (-n_fold)).reshape(-1, 1)
    reg_fold = LinearRegression().fit(X_tr_fold, y_tr)
    # Predict test point using fold-specific exponent
    X_te_fold = (d_te ** (-n_fold)).reshape(-1, 1)
    preds_opt_unbiased[te_idx] = reg_fold.predict(X_te_fold)
r2_opt_unbiased = r2_score(y, preds_opt_unbiased)
rmse_opt_unbiased = np.sqrt(mean_squared_error(y, preds_opt_unbiased))
print(f"  Optimized exponent unbiased LOO R²={r2_opt_unbiased:.4f} "
      f"(vs biased LOO R²={[r['LOO_R2'] for r in scaling_results if 'Optimized' in r['Scaling']][0]:.4f})")

# Override the 2-param version entry with 3-param version (correct k and IC)
# Also replace biased LOO with unbiased LOO
for r in scaling_results:
    if 'Optimized' in r['Scaling']:
        r['k'] = 3
        r['AIC'] = ic_nopt['AIC']
        r['AICc'] = ic_nopt['AICc']
        r['BIC'] = ic_nopt['BIC']
        r['LOO_R2'] = r2_opt_unbiased
        r['LOO_RMSE'] = rmse_opt_unbiased

# Sort by BIC
res_df = pd.DataFrame(scaling_results).sort_values('BIC')
print(f"\n  {'Scaling':<32s} {'k':>2s} {'Train R²':>8s} {'LOO R²':>7s} {'RMSE':>6s} "
      f"{'AIC':>8s} {'AICc':>8s} {'BIC':>8s}")
print(f"  {'-'*100}")
for _, r in res_df.iterrows():
    print(f"  {r['Scaling']:<32s} {r['k']:>2.0f} {r['Train_R2']:>8.4f} {r['LOO_R2']:>7.4f} "
          f"{r['LOO_RMSE']:>6.1f} {r['AIC']:>8.2f} {r['AICc']:>8.2f} {r['BIC']:>8.2f}")

best_scaling = res_df.iloc[0]
print(f"\n  ★ Best by BIC: {best_scaling['Scaling']} (BIC={best_scaling['BIC']:.2f})")

# Delta-AIC / Delta-BIC analysis
min_aic = res_df['AIC'].min()
min_bic = res_df['BIC'].min()
res_df['ΔAIC'] = res_df['AIC'] - min_aic
res_df['ΔBIC'] = res_df['BIC'] - min_bic
print(f"\n  ΔAIC / ΔBIC (lower = better, >10 = essentially no support):")
for _, r in res_df.iterrows():
    support = "strong" if r['ΔBIC'] < 2 else "moderate" if r['ΔBIC'] < 6 else "weak" if r['ΔBIC'] < 10 else "none"
    print(f"    {r['Scaling']:<32s} ΔAIC={r['ΔAIC']:>5.1f}  ΔBIC={r['ΔBIC']:>5.1f}  ({support})")

# ============================================================
# 4. PHASE 1b: COMPOSITION-DEPENDENT MODELS WITH SCALING LAWS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1b: COMPOSITION + GRAIN-SIZE SCALING (OLS with AIC/BIC)")
print("=" * 70)
print("σ_y = σ_0(comp, process) + k·f(d)")

comp_cols = [f'{el}_frac' for el in ELEMENTS]
proc_cols = ['ColdWork', 'RecrystT', 'HoldTime']
X_base = df_ys[comp_cols + proc_cols].fillna(0).values
k_base = X_base.shape[1] + 1  # +1 for intercept

# Add descriptors
desc_cols = ['delta', 'VEC', 'dH_mix', 'dS_mix', 'Omega', 'mu_bar',
             'delta_chi', 'Tm_bar', 'Phi_VLC', 'eps_Labusch',
             'sigma_y0_VLC', 'sigma_Labusch', 'sigma_TC', 'delta_Yang']
X_full_base = df_ys[comp_cols + proc_cols + desc_cols].fillna(0).values
k_full_base = X_full_base.shape[1] + 1

gs_scalings = {
    'd^(-1/2)':       d ** (-0.5),
    'd^(-1)':         d ** (-1.0),
    'd^(-1/3)':       d ** (-1.0/3.0),
    'd^(-2/3)':       d ** (-2.0/3.0),
    'ln(d)/d':        np.log(d) / d,
    f'd^(-{n_opt:.3f})': d ** (-n_opt),
    'd^(-1/2)+d^(-1)': np.column_stack([d**(-0.5), d**(-1.0)]),
}

comp_results = []
for gs_name, gs_feat in gs_scalings.items():
    if gs_feat.ndim == 1:
        gs_feat = gs_feat.reshape(-1, 1)
    n_gs = gs_feat.shape[1]

    for base_name, X_b, k_b in [('Comp+Proc', X_base, k_base),
                                  ('Comp+Proc+Desc', X_full_base, k_full_base)]:
        X_full = np.column_stack([X_b, gs_feat])
        k_total = k_b + n_gs

        reg = LinearRegression().fit(X_full, y)
        y_pred = reg.predict(X_full)
        ic = compute_ic(y, y_pred, k_total, n)

        r2_loo, rmse_loo, _ = loo_r2(X_full, y, LinearRegression)

        comp_results.append({
            'GS_Scaling': gs_name,
            'Base': base_name,
            'k': k_total,
            'Train_R2': r2_score(y, y_pred),
            'LOO_R2': r2_loo,
            'LOO_RMSE': rmse_loo,
            'AIC': ic['AIC'],
            'AICc': ic['AICc'],
            'BIC': ic['BIC'],
        })

comp_df = pd.DataFrame(comp_results).sort_values('BIC')
print(f"\n  {'GS Scaling':<18s} {'Base':<16s} {'k':>3s} {'Train R²':>8s} {'LOO R²':>7s} "
      f"{'RMSE':>6s} {'AIC':>8s} {'BIC':>8s}")
print(f"  {'-'*90}")
for _, r in comp_df.iterrows():
    print(f"  {r['GS_Scaling']:<18s} {r['Base']:<16s} {r['k']:>3.0f} {r['Train_R2']:>8.4f} "
          f"{r['LOO_R2']:>7.4f} {r['LOO_RMSE']:>6.1f} {r['AIC']:>8.2f} {r['BIC']:>8.2f}")

# ============================================================
# 5. ADD ALTERNATIVE GS FEATURES TO DATASET
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: EXTENDED FEATURE ENGINEERING")
print("=" * 70)

# Ensure d_inv_sqrt exists (do not rely on CSV pre-computing it)
df_ys['d_inv_sqrt'] = d ** (-0.5)

# New grain-size features
df_ys['d_inv'] = d ** (-1.0)                    # Dunstan-Bushby
df_ys['d_inv_third'] = d ** (-1.0/3.0)          # Baldwin
df_ys['d_inv_twothirds'] = d ** (-2.0/3.0)
df_ys['lnd_over_d'] = np.log(d) / d             # Critical thickness
df_ys['d_inv_nopt'] = d ** (-n_opt)              # Optimized
df_ys['ln_d'] = np.log(d)

# Interaction terms with alternative scalings
for el in ELEMENTS:
    df_ys[f'{el}_x_dinv'] = df_ys[f'{el}_frac'] * df_ys['d_inv_sqrt']
    df_ys[f'{el}_x_dinv1'] = df_ys[f'{el}_frac'] * df_ys['d_inv']

# SSS × alternative scalings
df_ys['VLC_x_dinv'] = df_ys['sigma_y0_VLC'] * df_ys['d_inv_sqrt']
df_ys['Labusch_x_dinv'] = df_ys['sigma_Labusch'] * df_ys['d_inv_sqrt']
df_ys['TC_x_dinv'] = df_ys['sigma_TC'] * df_ys['d_inv_sqrt']
df_ys['VLC_x_dinv1'] = df_ys['sigma_y0_VLC'] * df_ys['d_inv']
df_ys['Labusch_x_dinv1'] = df_ys['sigma_Labusch'] * df_ys['d_inv']

# Processing × grain interactions
df_ys['RecrystT_x_dinv'] = df_ys['RecrystT'] * df_ys['d_inv_sqrt']
df_ys['CW_x_dinv'] = df_ys['ColdWork'] * df_ys['d_inv_sqrt']
df_ys['RecrystT_x_dinv1'] = df_ys['RecrystT'] * df_ys['d_inv']
df_ys['CW_x_dinv1'] = df_ys['ColdWork'] * df_ys['d_inv']

# Quadratic/combined
df_ys['dinv_sq'] = df_ys['d_inv_sqrt'] ** 2
df_ys['V_frac_sq'] = df_ys['V_frac'] ** 2

# Descriptor ratios
df_ys['VEC_over_delta'] = df_ys['VEC'] / (df_ys['delta'] + 1e-8)
df_ys['mu_x_delta'] = df_ys['mu_bar'] * df_ys['delta']
df_ys['dH_over_Tm'] = df_ys['dH_mix'] / df_ys['Tm_bar']
df_ys['Omega_x_delta'] = df_ys['Omega'] * df_ys['delta']

# ============================================================
# 6. DEFINE FEATURE SETS
# ============================================================
GS_ALT = ['d_inv', 'd_inv_third', 'd_inv_twothirds', 'lnd_over_d', 'd_inv_nopt', 'ln_d']

FEAT_CORE = [f'{el}_frac' for el in ELEMENTS] + \
            ['d_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime']

FEAT_CORE_ALT = FEAT_CORE + GS_ALT

FEAT_PHYSICS = FEAT_CORE + [
    'delta', 'VEC', 'dH_mix', 'dS_mix', 'Omega', 'mu_bar', 'delta_chi',
    'Tm_bar', 'Phi_VLC', 'eps_Labusch', 'a_bar',
    'sigma_y0_VLC', 'sigma_Labusch', 'sigma_TC', 'delta_Yang',
]

FEAT_PHYSICS_ALT = FEAT_PHYSICS + GS_ALT

FEAT_INTERACTIONS = FEAT_PHYSICS + [
    f'{el}_x_dinv' for el in ELEMENTS
] + [
    'VEC_over_delta', 'mu_x_delta', 'dH_over_Tm', 'Omega_x_delta',
    'VLC_x_dinv', 'Labusch_x_dinv', 'TC_x_dinv',
    'RecrystT_x_dinv', 'CW_x_dinv', 'dinv_sq', 'V_frac_sq',
]

FEAT_INTERACTIONS_ALT = FEAT_INTERACTIONS + GS_ALT + [
    f'{el}_x_dinv1' for el in ELEMENTS
] + ['VLC_x_dinv1', 'Labusch_x_dinv1', 'RecrystT_x_dinv1', 'CW_x_dinv1']

# Build COMPACT via mutual information (from the extended set)
X_all_feat = df_ys[FEAT_INTERACTIONS_ALT].fillna(0).values
mi_scores = mutual_info_regression(X_all_feat, y, random_state=42)
mi_ranking = pd.Series(mi_scores, index=FEAT_INTERACTIONS_ALT).sort_values(ascending=False)
FEAT_COMPACT = mi_ranking.head(20).index.tolist()

FEAT_MAP = {
    'CORE': FEAT_CORE,
    'CORE_ALT': FEAT_CORE_ALT,
    'PHYSICS': FEAT_PHYSICS,
    'PHYSICS_ALT': FEAT_PHYSICS_ALT,
    'INTERACTIONS': FEAT_INTERACTIONS,
    'INTERACTIONS_ALT': FEAT_INTERACTIONS_ALT,
    'COMPACT': FEAT_COMPACT,
}

for k_name, v in FEAT_MAP.items():
    print(f"  {k_name}: {len(v)} features")
print(f"  COMPACT (top 20 by MI): {FEAT_COMPACT}")


def get_Xy(feat_set):
    X = df_ys[feat_set].fillna(0).values
    return X, y, feat_set


# ============================================================
# 7. EVALUATION FUNCTIONS
# ============================================================
def eval_loo(model_factory, X, y):
    """LOO cross-validation returning R², RMSE, MAE, and predictions."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    return r2, rmse, mae, preds


def eval_lobo(model_factory, X, y, groups):
    """Leave-one-batch-out cross-validation."""
    logo = LeaveOneGroupOut()
    preds = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds)


def run_optuna(model_class, space_fn, X, y, n_trials=50,
               needs_scaling=False, timeout=300):
    """Optuna HPO with repeated k-fold CV."""
    def objective(trial):
        params = space_fn(trial)
        if needs_scaling:
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('model', model_class(**params))])
        else:
            pipe = model_class(**params)
        scores = cross_val_score(pipe, X, y,
                                  cv=RepeatedKFold(n_splits=5, n_repeats=3,
                                                   random_state=42),
                                  scoring='neg_mean_squared_error')
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False)
    return study.best_params, -study.best_value


def count_effective_params(model, model_name, X):
    """Estimate effective number of parameters for AIC/BIC.

    For Ridge: k_eff = trace(H) = sum(d_i^2 / (d_i^2 + alpha)) + 1 (intercept),
    where d_i are singular values of X (centered if intercept is fit separately).
    For Lasso/ElasticNet: count of non-zero coefficients + 1 (intercept).
    For tree ensembles: approximate via leaf counts (see reliable_ic flag).
    """
    n_feat = X.shape[1]
    if 'OLS' in model_name or 'Baseline' in model_name:
        return n_feat + 1
    elif 'Ridge' in model_name and 'Lasso' not in model_name and 'ElasticNet' not in model_name:
        # Ridge: compute effective degrees of freedom via SVD
        # k_eff = sum(d_i^2 / (d_i^2 + alpha)) + 1 for intercept
        alpha = None
        if hasattr(model, 'alpha_'):
            alpha = model.alpha_  # RidgeCV
        elif hasattr(model, 'alpha'):
            alpha = model.alpha   # Ridge
        if alpha is not None and alpha > 0:
            # Center X for SVD (intercept is fit separately by sklearn Ridge)
            X_c = X - X.mean(axis=0)
            _, s, _ = np.linalg.svd(X_c, full_matrices=False)
            k_eff = np.sum(s**2 / (s**2 + alpha)) + 1  # +1 for intercept
            return k_eff
        return n_feat + 1
    elif 'ElasticNet' in model_name or 'Lasso' in model_name:
        # Lasso/ElasticNet: count non-zero coefficients + 1 (intercept)
        if hasattr(model, 'coef_'):
            return np.sum(np.abs(model.coef_) > 1e-10) + 1
        return n_feat + 1
    elif 'SVR' in model_name:
        if hasattr(model, 'named_steps'):
            svr = model.named_steps['model']
        else:
            svr = model
        return getattr(svr, 'n_support_', [n_feat])[0] if hasattr(svr, 'n_support_') else n_feat
    elif 'GPR' in model_name:
        # GPR: effective params ≈ n (interpolating model)
        return X.shape[0]
    elif any(t in model_name for t in ['XGBoost', 'CatBoost', 'LightGBM', 'RF', 'Random Forest']):
        # Tree models: approximate by number of leaves
        if hasattr(model, 'get_booster'):
            try:
                trees = model.get_booster().get_dump()
                n_leaves = sum(t.count('leaf') for t in trees)
                return n_leaves
            except:
                pass
        if hasattr(model, 'tree_count_'):
            return model.tree_count_ * 4  # rough approx
        return n_feat * 5  # rough fallback
    elif 'KRR' in model_name:
        return X.shape[0]  # kernel model, n effective params
    return n_feat + 1


# ============================================================
# 8. MODEL DEFINITIONS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: EXHAUSTIVE MODEL SEARCH WITH INFORMATION CRITERIA")
print("=" * 70)

# Optuna search spaces
def ridge_space(trial):
    return {'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True)}

def elasticnet_space(trial):
    return {'alpha': trial.suggest_float('alpha', 1e-3, 1e2, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 0.99)}

def svr_space(trial):
    return {'C': trial.suggest_float('C', 0.1, 1000, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.1, 50, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])}

def krr_space(trial):
    return {'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'kernel': 'rbf'}

def rf_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 20),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42}

def xgb_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'random_state': 42, 'verbosity': 0}

def catboost_space(trial):
    return {'iterations': trial.suggest_int('iterations', 100, 600, step=50),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.01, 2.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 128),
            'random_seed': 42, 'verbose': 0}

def lgbm_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 60),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'random_state': 42, 'verbosity': -1}

# Model configurations: name -> (class, space_fn, feature_set, needs_scaling)
MODELS = {
    # --- No HPO ---
    'Baseline (mean)': {
        'cls': DummyRegressor, 'space': None, 'feat': 'CORE', 'scale': False,
        'factory_kw': {'strategy': 'mean'},
    },
    'OLS': {
        'cls': LinearRegression, 'space': None, 'feat': 'PHYSICS', 'scale': False,
        'factory_kw': {},
    },
    'OLS (alt-GS)': {
        'cls': LinearRegression, 'space': None, 'feat': 'PHYSICS_ALT', 'scale': False,
        'factory_kw': {},
    },
    'GPR (Matern)': {
        'cls': None, 'space': None, 'feat': 'COMPACT', 'scale': True,
        'factory_kw': {},
    },
    # --- Optuna-tuned ---
    'Ridge': {
        'cls': Ridge, 'space': ridge_space, 'feat': 'PHYSICS_ALT', 'scale': True,
        'factory_kw': {},
    },
    'ElasticNet': {
        'cls': ElasticNet, 'space': elasticnet_space, 'feat': 'PHYSICS_ALT', 'scale': True,
        'factory_kw': {},
    },
    'SVR (RBF)': {
        'cls': SVR, 'space': svr_space, 'feat': 'COMPACT', 'scale': True,
        'factory_kw': {},
    },
    'KRR (RBF)': {
        'cls': KernelRidge, 'space': krr_space, 'feat': 'COMPACT', 'scale': True,
        'factory_kw': {},
    },
    'Random Forest': {
        'cls': RandomForestRegressor, 'space': rf_space, 'feat': 'INTERACTIONS_ALT',
        'scale': False, 'factory_kw': {},
    },
    'XGBoost': {
        'cls': xgb.XGBRegressor, 'space': xgb_space, 'feat': 'INTERACTIONS_ALT',
        'scale': False, 'factory_kw': {},
    },
    'CatBoost': {
        'cls': cb.CatBoostRegressor, 'space': catboost_space, 'feat': 'INTERACTIONS_ALT',
        'scale': False, 'factory_kw': {},
    },
    'LightGBM': {
        'cls': lgb.LGBMRegressor, 'space': lgbm_space, 'feat': 'INTERACTIONS_ALT',
        'scale': False, 'factory_kw': {},
    },
    # --- Compact variants ---
    'XGBoost-compact': {
        'cls': xgb.XGBRegressor, 'space': xgb_space, 'feat': 'COMPACT',
        'scale': False, 'factory_kw': {},
    },
    'CatBoost-compact': {
        'cls': cb.CatBoostRegressor, 'space': catboost_space, 'feat': 'COMPACT',
        'scale': False, 'factory_kw': {},
    },
}

# ============================================================
# 9. RUN ALL MODELS
# ============================================================
results = []
all_loo_preds = {}

for model_name, cfg in MODELS.items():
    feat_key = cfg['feat']
    X_m, y_m, cols_m = get_Xy(FEAT_MAP[feat_key])
    t0 = time.time()

    # --- GPR (special) ---
    if model_name == 'GPR (Matern)':
        print(f"\n  [{model_name}] features={feat_key} ({X_m.shape[1]}), n={n}...")
        kernel = ConstantKernel(100.0) * Matern(length_scale=np.ones(X_m.shape[1]),
                                                  nu=2.5) + WhiteKernel(1.0)
        def gpr_factory():
            return Pipeline([('scaler', StandardScaler()),
                             ('model', GaussianProcessRegressor(
                                 kernel=kernel, n_restarts_optimizer=5,
                                 alpha=1e-6, random_state=42))])
        r2_l, rmse_l, mae_l, preds_l = eval_loo(gpr_factory, X_m, y_m)
        lobo_r2 = eval_lobo(gpr_factory, X_m, y_m, groups)
        dt = time.time() - t0
        # Train for AIC/BIC
        gpr_full = gpr_factory()
        gpr_full.fit(X_m, y_m)
        y_train_pred = gpr_full.predict(X_m)
        k_eff = count_effective_params(gpr_full, model_name, X_m)
        ic = compute_ic(y_m, y_train_pred, k_eff, n)
        results.append({
            'Model': model_name, 'Features': feat_key, 'n_feat': X_m.shape[1],
            'LOO_R2': r2_l, 'LOO_RMSE': rmse_l, 'LOO_MAE': mae_l,
            'LOBO_R2': lobo_r2, 'Train_R2': r2_score(y_m, y_train_pred),
            'k_eff': k_eff, 'AIC': ic['AIC'], 'AICc': ic['AICc'], 'BIC': ic['BIC'],
            'HPO': 'None', 'time': dt,
        })
        all_loo_preds[model_name] = preds_l
        print(f"    LOO R²={r2_l:.4f}, RMSE={rmse_l:.1f} | LOBO R²={lobo_r2:.4f} | "
              f"AIC={ic['AIC']:.1f}, BIC={ic['BIC']:.1f} | {dt:.1f}s")
        continue

    # --- No HPO models ---
    if cfg['space'] is None:
        print(f"\n  [{model_name}] features={feat_key} ({X_m.shape[1]}), n={n}...")
        kw = cfg['factory_kw']
        if cfg['scale']:
            factory = lambda: Pipeline([('scaler', StandardScaler()),
                                         ('model', cfg['cls'](**kw))])
        else:
            factory = lambda kw=kw: cfg['cls'](**kw)
        r2_l, rmse_l, mae_l, preds_l = eval_loo(factory, X_m, y_m)
        lobo_r2 = eval_lobo(factory, X_m, y_m, groups)
        dt = time.time() - t0
        # Train for AIC/BIC
        full_model = factory()
        full_model.fit(X_m, y_m)
        y_train_pred = full_model.predict(X_m)
        k_eff = count_effective_params(full_model, model_name, X_m)
        ic = compute_ic(y_m, y_train_pred, k_eff, n)
        results.append({
            'Model': model_name, 'Features': feat_key, 'n_feat': X_m.shape[1],
            'LOO_R2': r2_l, 'LOO_RMSE': rmse_l, 'LOO_MAE': mae_l,
            'LOBO_R2': lobo_r2, 'Train_R2': r2_score(y_m, y_train_pred),
            'k_eff': k_eff, 'AIC': ic['AIC'], 'AICc': ic['AICc'], 'BIC': ic['BIC'],
            'HPO': 'None', 'time': dt,
        })
        all_loo_preds[model_name] = preds_l
        print(f"    LOO R²={r2_l:.4f}, RMSE={rmse_l:.1f} | LOBO R²={lobo_r2:.4f} | "
              f"AIC={ic['AIC']:.1f}, BIC={ic['BIC']:.1f} | {dt:.1f}s")
        continue

    # --- Optuna-tuned models ---
    print(f"\n  [{model_name}] features={feat_key} ({X_m.shape[1]}), Optuna 50 trials...")
    try:
        best_params, best_cv_mse = run_optuna(
            cfg['cls'], cfg['space'], X_m, y_m,
            n_trials=50, needs_scaling=cfg['scale'], timeout=300)
        print(f"    Best params: { {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()} }")

        if cfg['scale']:
            factory = lambda bp=best_params: Pipeline([
                ('scaler', StandardScaler()),
                ('model', cfg['cls'](**bp))])
        else:
            factory = lambda bp=best_params: cfg['cls'](**bp)

        r2_l, rmse_l, mae_l, preds_l = eval_loo(factory, X_m, y_m)
        lobo_r2 = eval_lobo(factory, X_m, y_m, groups)
        dt = time.time() - t0

        # Train for AIC/BIC
        full_model = factory()
        full_model.fit(X_m, y_m)
        y_train_pred = full_model.predict(X_m)
        if cfg['scale']:
            inner = full_model.named_steps['model']
            # Pass scaled X for correct SVD-based k_eff (Ridge)
            X_for_keff = full_model.named_steps['scaler'].transform(X_m)
        else:
            inner = full_model
            X_for_keff = X_m
        k_eff = count_effective_params(inner, model_name, X_for_keff)
        ic = compute_ic(y_m, y_train_pred, k_eff, n)

        results.append({
            'Model': model_name, 'Features': feat_key, 'n_feat': X_m.shape[1],
            'LOO_R2': r2_l, 'LOO_RMSE': rmse_l, 'LOO_MAE': mae_l,
            'LOBO_R2': lobo_r2, 'Train_R2': r2_score(y_m, y_train_pred),
            'k_eff': k_eff, 'AIC': ic['AIC'], 'AICc': ic['AICc'], 'BIC': ic['BIC'],
            'HPO': 'Optuna-50', 'time': dt,
        })
        all_loo_preds[model_name] = preds_l
        print(f"    LOO R²={r2_l:.4f}, RMSE={rmse_l:.1f} | LOBO R²={lobo_r2:.4f} | "
              f"AIC={ic['AIC']:.1f}, BIC={ic['BIC']:.1f} | {dt:.1f}s")

    except Exception as e:
        print(f"    FAILED: {e}")
        results.append({
            'Model': model_name, 'Features': feat_key, 'n_feat': X_m.shape[1],
            'LOO_R2': np.nan, 'LOO_RMSE': np.nan, 'LOO_MAE': np.nan,
            'LOBO_R2': np.nan, 'Train_R2': np.nan,
            'k_eff': np.nan, 'AIC': np.nan, 'AICc': np.nan, 'BIC': np.nan,
            'HPO': 'FAILED', 'time': 0,
        })

# ============================================================
# 10. PHYSICS-INFORMED BASE MODEL (M3) + STACKING ENSEMBLE
# ============================================================
print("\n" + "=" * 70)
print("PHYSICS-INFORMED BASE MODEL + STACKING ENSEMBLE")
print("=" * 70)

# --- Build M3: σ₀(comp) + k·d⁻¹/² LOO predictions ---
# This is the composition-dependent Hall-Petch model from composition_hp_analysis.py.
# NOTE: We drop Ni as the reference element to avoid rank deficiency.
# The 8 element fractions sum to 1, which is collinear with the intercept column.
# Using 7 elements + intercept gives unique, interpretable coefficients (each coeff
# represents the effect of substituting that element for Ni). The PREDICTIONS are
# identical to the 8-element formulation — only the coefficient decomposition changes.
elem_names_m3 = [f'{el}_frac' for el in ELEMENTS if el != 'Ni']
d_inv_sqrt_m3 = d ** (-0.5)
X_m3 = np.column_stack([np.ones(n), df_ys[elem_names_m3].values, d_inv_sqrt_m3])
k_m3 = X_m3.shape[1]  # 9 columns: intercept + 7 elements (Ni ref) + d^(-1/2)

# fit_intercept=False because X_m3 already contains an explicit ones column
m3_preds = np.zeros(n)
loo = LeaveOneOut()
for tr, te in loo.split(X_m3):
    reg = LinearRegression(fit_intercept=False).fit(X_m3[tr], y[tr])
    m3_preds[te] = reg.predict(X_m3[te])
m3_r2 = r2_score(y, m3_preds)
m3_rmse = np.sqrt(mean_squared_error(y, m3_preds))
m3_mae = mean_absolute_error(y, m3_preds)

# LOBO for M3
logo_m3 = LeaveOneGroupOut()
m3_lobo_preds = np.zeros(n)
for tr, te in logo_m3.split(X_m3, y, groups):
    reg = LinearRegression(fit_intercept=False).fit(X_m3[tr], y[tr])
    m3_lobo_preds[te] = reg.predict(X_m3[te])
m3_lobo_r2 = r2_score(y, m3_lobo_preds)

# AIC/BIC for M3
reg_m3_full = LinearRegression(fit_intercept=False).fit(X_m3, y)
y_m3_train = reg_m3_full.predict(X_m3)
ic_m3 = compute_ic(y, y_m3_train, k_m3, n)

all_loo_preds['M3: σ₀(all elem)'] = m3_preds
results.append({
    'Model': 'M3: σ₀(all elem)', 'Features': 'HP_PHYSICS',
    'n_feat': k_m3 - 1,
    'LOO_R2': m3_r2, 'LOO_RMSE': m3_rmse, 'LOO_MAE': m3_mae,
    'LOBO_R2': m3_lobo_r2, 'Train_R2': r2_score(y, y_m3_train),
    'k_eff': k_m3, 'AIC': ic_m3['AIC'],
    'AICc': ic_m3['AICc'], 'BIC': ic_m3['BIC'],
    'HPO': 'None (OLS)', 'time': 0,
})
print(f"  M3: σ₀(all elem) LOO R²={m3_r2:.4f}, RMSE={m3_rmse:.1f}, LOBO R²={m3_lobo_r2:.4f}")
print(f"    AIC={ic_m3['AIC']:.1f}, BIC={ic_m3['BIC']:.1f}, k_eff={k_m3}")

res_df = pd.DataFrame(results).sort_values('LOO_R2', ascending=False)
valid = res_df[res_df['LOO_R2'].notna() & (res_df['LOO_R2'] > 0)].copy()

# Pick diverse base learners (now includes physics-informed family)
categories = {
    'tree': ['XGBoost', 'CatBoost', 'LightGBM', 'Random Forest'],
    'kernel': ['SVR (RBF)', 'GPR (Matern)', 'KRR (RBF)'],
    'linear': ['ElasticNet', 'Ridge', 'OLS', 'OLS (alt-GS)'],
    'boosting_compact': ['XGBoost-compact', 'CatBoost-compact'],
    'physics': ['M3: σ₀(all elem)'],
}

stack_models = []
for cat, names in categories.items():
    for nm in names:
        if nm in all_loo_preds:
            stack_models.append(nm)
            break

print(f"Stacking {len(stack_models)} models: {stack_models}")

if len(stack_models) >= 3:
    meta_X = np.column_stack([all_loo_preds[m] for m in stack_models])

    # Average ensemble
    avg_pred = meta_X.mean(axis=1)
    avg_r2 = r2_score(y, avg_pred)
    avg_rmse = np.sqrt(mean_squared_error(y, avg_pred))
    avg_mae = mean_absolute_error(y, avg_pred)
    # NOTE: Average ensemble IC uses LOO predictions (not training predictions),
    # so these ICs are NOT directly comparable to base model ICs which use
    # in-sample training predictions. They appear artificially favorable because
    # LOO residuals are used in place of training residuals.
    ic_avg = compute_ic(y, avg_pred, len(stack_models), n)
    print(f"  Average ensemble LOO: R²={avg_r2:.4f}, RMSE={avg_rmse:.1f}")
    print(f"    AIC={ic_avg['AIC']:.1f}, BIC={ic_avg['BIC']:.1f} (based on LOO preds, not training preds)")
    results.append({
        'Model': 'Average Ensemble', 'Features': 'meta',
        'n_feat': len(stack_models),
        'LOO_R2': avg_r2, 'LOO_RMSE': avg_rmse, 'LOO_MAE': avg_mae,
        'LOBO_R2': np.nan, 'Train_R2': np.nan,
        'k_eff': len(stack_models), 'AIC': ic_avg['AIC'],
        'AICc': ic_avg['AICc'], 'BIC': ic_avg['BIC'],
        'HPO': 'None', 'time': 0,
    })

    # Stacking with Ridge meta-learner (LOO on meta)
    loo = LeaveOneOut()
    stack_preds = np.zeros(n)
    for tr, te in loo.split(meta_X):
        ridge_meta = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge_meta.fit(meta_X[tr], y[tr])
        stack_preds[te] = ridge_meta.predict(meta_X[te])
    stack_r2 = r2_score(y, stack_preds)
    stack_rmse = np.sqrt(mean_squared_error(y, stack_preds))
    stack_mae = mean_absolute_error(y, stack_preds)

    # LOBO for stacking
    logo = LeaveOneGroupOut()
    stack_lobo_preds = np.zeros(n)
    for tr, te in logo.split(meta_X, y, groups):
        ridge_meta = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge_meta.fit(meta_X[tr], y[tr])
        stack_lobo_preds[te] = ridge_meta.predict(meta_X[te])
    stack_lobo_r2 = r2_score(y, stack_lobo_preds)

    # AIC/BIC for stacking (fit on full meta_X)
    # NOTE: This k_eff only captures meta-learner complexity (Ridge over base
    # model predictions). It does NOT account for the total complexity of the
    # base learners themselves. The stacking IC is therefore a lower bound on
    # true model complexity and should be interpreted cautiously vs. base models.
    ridge_full = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(meta_X, y)
    y_stack_train = ridge_full.predict(meta_X)
    # Use SVD-based effective df for Ridge meta-learner
    k_stack = count_effective_params(ridge_full, 'Ridge-meta', meta_X)
    ic_stack = compute_ic(y, y_stack_train, k_stack, n)

    print(f"  Stacking (Ridge) LOO: R²={stack_r2:.4f}, RMSE={stack_rmse:.1f}")
    print(f"    LOBO R²={stack_lobo_r2:.4f} | AIC={ic_stack['AIC']:.1f}, BIC={ic_stack['BIC']:.1f}")
    results.append({
        'Model': 'Stacking (Ridge)', 'Features': 'meta',
        'n_feat': len(stack_models),
        'LOO_R2': stack_r2, 'LOO_RMSE': stack_rmse, 'LOO_MAE': stack_mae,
        'LOBO_R2': stack_lobo_r2, 'Train_R2': r2_score(y, y_stack_train),
        'k_eff': k_stack, 'AIC': ic_stack['AIC'],
        'AICc': ic_stack['AICc'], 'BIC': ic_stack['BIC'],
        'HPO': 'RidgeCV', 'time': 0,
    })

# ============================================================
# 11. FINAL RESULTS TABLE
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

res_df = pd.DataFrame(results).sort_values('LOO_R2', ascending=False).reset_index(drop=True)

# Add reliable_ic flag: AIC/BIC are well-defined for parametric models (OLS, Ridge,
# Lasso, ElasticNet, M3). For tree ensembles, kernel methods, and stacking, k_eff is
# approximate, so AIC/BIC should be interpreted cautiously vs. parametric models.
def _is_reliable_ic(model_name):
    reliable_names = ['OLS', 'Ridge', 'ElasticNet', 'Lasso', 'M3', 'Baseline']
    # Stacking/Average ensembles are not reliable (meta-learner k only)
    if 'Ensemble' in model_name or 'Stacking' in model_name:
        return False
    return any(tag in model_name for tag in reliable_names)

res_df['reliable_ic'] = res_df['Model'].apply(_is_reliable_ic)

print(f"\n{'Rank':>4s}  {'Model':<24s} {'Feat':<18s} {'nF':>3s} {'k_eff':>5s} "
      f"{'LOO R²':>7s} {'RMSE':>6s} {'LOBO R²':>8s} "
      f"{'AIC':>8s} {'AICc':>8s} {'BIC':>8s} {'HPO':<12s} {'Time':>5s}")
print(f"  {'-'*140}")

for i, r in res_df.iterrows():
    loo_str = f"{r['LOO_R2']:.4f}" if pd.notna(r['LOO_R2']) else "N/A"
    lobo_str = f"{r['LOBO_R2']:.4f}" if pd.notna(r['LOBO_R2']) else "N/A"
    aic_str = f"{r['AIC']:.1f}" if pd.notna(r['AIC']) else "N/A"
    aicc_str = f"{r['AICc']:.1f}" if pd.notna(r['AICc']) else "N/A"
    bic_str = f"{r['BIC']:.1f}" if pd.notna(r['BIC']) else "N/A"
    keff_str = f"{r['k_eff']:.1f}" if pd.notna(r['k_eff']) else "N/A"
    rmse_str = f"{r['LOO_RMSE']:.1f}" if pd.notna(r['LOO_RMSE']) else "N/A"
    print(f"{i+1:>4d}. {r['Model']:<24s} {r['Features']:<18s} {r['n_feat']:>3.0f} "
          f"{keff_str:>5s} {loo_str:>7s} {rmse_str:>6s} {lobo_str:>8s} "
          f"{aic_str:>8s} {aicc_str:>8s} {bic_str:>8s} {r['HPO']:<12s} {r['time']:>5.0f}s")

# ΔAIC / ΔBIC for models with valid AIC
valid_aic = res_df[res_df['AIC'].notna()].copy()
if len(valid_aic) > 0:
    min_aic = valid_aic['AIC'].min()
    min_bic = valid_aic['BIC'].min()
    valid_aic['ΔAIC'] = valid_aic['AIC'] - min_aic
    valid_aic['ΔBIC'] = valid_aic['BIC'] - min_bic

    print(f"\n  Information Criteria Ranking (ΔBIC, lower = better):")
    print(f"  NOTE: Entries marked with ~ use approximate k_eff (tree/kernel/ensemble)")
    print(f"        and should be compared cautiously against parametric models.")
    for _, r in valid_aic.sort_values('BIC').iterrows():
        support = ("strong" if r['ΔBIC'] < 2 else "moderate" if r['ΔBIC'] < 6
                    else "weak" if r['ΔBIC'] < 10 else "none")
        ic_marker = " " if r.get('reliable_ic', True) else "~"
        keff_fmt = f"{r['k_eff']:>5.0f}" if pd.notna(r['k_eff']) else "  N/A"
        print(f"  {ic_marker} {r['Model']:<24s} ΔAIC={r['ΔAIC']:>6.1f}  ΔBIC={r['ΔBIC']:>6.1f}  "
              f"k_eff={keff_fmt}  ({support} support)")

# Save to CSV
res_df.to_csv(f'{RESULTS_DIR}/model_search_results_v2.csv', index=False)
print(f"\nResults saved to model_search_results_v2.csv")

# ============================================================
# 12. VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# --- Plot 25: Grain-size scaling comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) Scaling fits
ax = axes[0]
d_plot = np.linspace(d.min(), d.max(), 200)
scale_df = pd.DataFrame(scaling_results).sort_values('BIC')
colors = plt.cm.tab10(np.linspace(0, 1, len(scale_df)))
for i, (_, r) in enumerate(scale_df.iterrows()):
    name = r['Scaling']
    ax.scatter(d, y, c='gray', s=15, alpha=0.3, zorder=1)
ax.set_xlabel('Grain Size d (µm)', fontsize=12)
ax.set_ylabel('Yield Strength (MPa)', fontsize=12)
ax.set_title('Grain-Size Scaling Comparison', fontsize=13)

# Plot top 4 fits
top4 = scale_df.head(4)
colors_top = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
for i, (_, r) in enumerate(top4.iterrows()):
    name = r['Scaling'].split('[')[0].strip()
    f_plot = None
    if 'd^(-1/2)' in r['Scaling'] and 'Composite' not in r['Scaling']:
        f_plot = d_plot ** (-0.5)
    elif 'd^(-1)' in r['Scaling'] and 'Composite' not in r['Scaling'] and 'Taylor' not in r['Scaling']:
        f_plot = d_plot ** (-1.0)
    elif 'd^(-1/3)' in r['Scaling']:
        f_plot = d_plot ** (-1.0/3.0)
    elif 'Optimized' in r['Scaling']:
        f_plot = d_plot ** (-n_opt)
    elif 'ln(d)/d' in r['Scaling']:
        f_plot = np.log(d_plot) / d_plot
    elif 'd^(-2/3)' in r['Scaling']:
        f_plot = d_plot ** (-2.0/3.0)
    if f_plot is not None:
        y_plot = r['intercept'] + r['slope'] * f_plot
        ax.plot(d_plot, y_plot, '-', color=colors_top[i], linewidth=2,
                label=f"{name} (BIC={r['BIC']:.0f})", zorder=2)
ax.scatter(d, y, c='black', s=20, alpha=0.6, zorder=3)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# (b) BIC comparison bar chart
ax = axes[1]
scale_df_plot = scale_df.copy()
min_bic_scale = scale_df_plot['BIC'].min()
scale_df_plot['ΔBIC'] = scale_df_plot['BIC'] - min_bic_scale
bars = ax.barh(range(len(scale_df_plot)), scale_df_plot['ΔBIC'].values,
               color=['#009E73' if v < 2 else '#E69F00' if v < 6 else '#D55E00'
                      for v in scale_df_plot['ΔBIC'].values])
ax.set_yticks(range(len(scale_df_plot)))
ax.set_yticklabels([s.split('[')[0].strip() for s in scale_df_plot['Scaling'].values], fontsize=9)
ax.set_xlabel('ΔBIC (lower = better)', fontsize=12)
ax.set_title('Grain-Size Scaling: ΔBIC', fontsize=13)
ax.axvline(2, color='green', linestyle='--', alpha=0.5, label='Strong support (<2)')
ax.axvline(6, color='orange', linestyle='--', alpha=0.5, label='Moderate (<6)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# (c) LOO R² vs BIC for all Phase 3 models
ax = axes[2]
valid_plot = res_df[res_df['LOO_R2'].notna() & (res_df['LOO_R2'] > -1) &
                     res_df['BIC'].notna()].copy()
cat_colors = {'linear': '#0072B2', 'tree': '#D55E00', 'kernel': '#009E73',
              'ensemble': '#CC79A7', 'baseline': '#95A5A6'}
for _, r in valid_plot.iterrows():
    m = r['Model']
    if m in ['OLS', 'OLS (alt-GS)', 'Ridge', 'ElasticNet']:
        c = cat_colors['linear']
    elif any(t in m for t in ['XGBoost', 'CatBoost', 'LightGBM', 'RF', 'Random Forest']):
        c = cat_colors['tree']
    elif any(t in m for t in ['GPR', 'SVR', 'KRR']):
        c = cat_colors['kernel']
    elif 'Ensemble' in m or 'Stacking' in m:
        c = cat_colors['ensemble']
    else:
        c = cat_colors['baseline']
    ax.scatter(r['BIC'], r['LOO_R2'], c=c, s=80, edgecolors='k', linewidth=0.5, zorder=3)
    ax.annotate(r['Model'], (r['BIC'], r['LOO_R2']),
                textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

# Legend
for cat, col in cat_colors.items():
    ax.scatter([], [], c=col, s=60, edgecolors='k', linewidth=0.5, label=cat)
ax.legend(fontsize=8, loc='lower left')
ax.set_xlabel('BIC (lower = better, penalizes complexity)', fontsize=11)
ax.set_ylabel('LOO R² (higher = better)', fontsize=11)
ax.set_title('Model Selection: LOO R² vs BIC', fontsize=13)
ax.grid(True, alpha=0.3)

plt.suptitle('Grain-Size Scaling & Information Criteria Analysis', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/25_scaling_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 25_scaling_comparison.png")

# --- Plot 26: Full model comparison (LOO R², LOBO R², BIC) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) LOO R² bar chart with AIC annotation
ax = axes[0]
plot_df = res_df[res_df['LOO_R2'].notna() & (res_df['LOO_R2'] > -1)].head(16).copy()
plot_df = plot_df.sort_values('LOO_R2', ascending=True)

colors_bar = []
for _, r in plot_df.iterrows():
    m = r['Model']
    if m in ['OLS', 'OLS (alt-GS)', 'Ridge', 'ElasticNet']:
        colors_bar.append('#0072B2')
    elif any(t in m for t in ['XGBoost', 'CatBoost', 'LightGBM', 'RF', 'Random Forest']):
        colors_bar.append('#D55E00')
    elif any(t in m for t in ['GPR', 'SVR', 'KRR']):
        colors_bar.append('#009E73')
    elif 'Ensemble' in m or 'Stacking' in m:
        colors_bar.append('#CC79A7')
    else:
        colors_bar.append('#95A5A6')

bars = ax.barh(range(len(plot_df)), plot_df['LOO_R2'].values, color=colors_bar,
               edgecolor='k', linewidth=0.5)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df['Model'].values, fontsize=9)
ax.set_xlabel('LOO R²', fontsize=12)
ax.set_title('Model Comparison: LOO R²', fontsize=13)
ax.axvline(0.644, color='gray', linestyle='--', alpha=0.7, label='Previous best (0.644)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# Annotate BIC
for i, (_, r) in enumerate(plot_df.iterrows()):
    if pd.notna(r['BIC']) and r['LOO_R2'] > 0:
        ax.annotate(f"BIC={r['BIC']:.0f}", (r['LOO_R2'], i),
                    textcoords="offset points", xytext=(5, 0), fontsize=7, alpha=0.7)

# (b) LOO R² vs LOBO R²
ax = axes[1]
valid_lobo = res_df[res_df['LOO_R2'].notna() & res_df['LOBO_R2'].notna() &
                     (res_df['LOO_R2'] > -1)].copy()
for _, r in valid_lobo.iterrows():
    m = r['Model']
    if m in ['OLS', 'OLS (alt-GS)', 'Ridge', 'ElasticNet']:
        c = '#0072B2'
    elif any(t in m for t in ['XGBoost', 'CatBoost', 'LightGBM', 'RF', 'Random Forest']):
        c = '#D55E00'
    elif any(t in m for t in ['GPR', 'SVR', 'KRR']):
        c = '#009E73'
    elif 'Ensemble' in m or 'Stacking' in m:
        c = '#CC79A7'
    else:
        c = '#95A5A6'
    ax.scatter(r['LOO_R2'], r['LOBO_R2'], c=c, s=100, edgecolors='k',
               linewidth=0.5, zorder=3)
    ax.annotate(r['Model'], (r['LOO_R2'], r['LOBO_R2']),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

lims = [min(valid_lobo['LOO_R2'].min(), valid_lobo['LOBO_R2'].min()) - 0.05,
        max(valid_lobo['LOO_R2'].max(), valid_lobo['LOBO_R2'].max()) + 0.05]
ax.plot(lims, lims, 'k--', alpha=0.4)
ax.set_xlabel('LOO R² (93 folds)', fontsize=12)
ax.set_ylabel('LOBO R² (6 batch folds)', fontsize=12)
ax.set_title('Generalization: LOO vs LOBO', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/26_model_ic_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 26_model_ic_comparison.png")

# --- Plot 27: Parity plots for top models ---
top_models = [m for m in res_df['Model'].values if m in all_loo_preds][:12]
n_top = len(top_models)
ncols = 4
nrows = (n_top + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes_flat = axes.flatten() if n_top > 4 else [axes] if n_top == 1 else axes.flatten()

batch_labels = df_ys['Iteration'].values
for i, model_name in enumerate(top_models):
    ax = axes_flat[i]
    preds = all_loo_preds[model_name]
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    for batch in sorted(BATCH_COLORS.keys()):
        mask = batch_labels == batch
        if mask.any():
            ax.scatter(y[mask], preds[mask], c=BATCH_COLORS[batch], s=30,
                       alpha=0.7, edgecolors='k', linewidth=0.3, label=batch)

    lims = [min(y.min(), preds.min()) * 0.9, max(y.max(), preds.max()) * 1.1]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f'{model_name}\nR²={r2:.3f}, RMSE={rmse:.0f}', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=6, loc='upper left')

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.supxlabel('Experimental YS (MPa)', fontsize=12)
fig.supylabel('Predicted YS (MPa)', fontsize=12)
plt.suptitle('LOO Parity Plots — Top Models', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/27_parity_grid_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 27_parity_grid_v2.png")

# --- Plot 28: SHAP for best tree model ---
best_tree = None
for m in res_df['Model'].values:
    if m in ['XGBoost', 'CatBoost', 'LightGBM', 'XGBoost-compact', 'CatBoost-compact']:
        best_tree = m
        break

if best_tree:
    print(f"\n  Computing SHAP for {best_tree}...")
    cfg = MODELS.get(best_tree)
    if cfg and cfg['space'] is not None:
        feat_key = cfg['feat']
        X_shap, y_shap, cols_shap = get_Xy(FEAT_MAP[feat_key])
        try:
            best_params_shap, _ = run_optuna(cfg['cls'], cfg['space'], X_shap, y_shap,
                                              n_trials=30, needs_scaling=False, timeout=120)
            model_shap = cfg['cls'](**best_params_shap)
            model_shap.fit(X_shap, y_shap)

            explainer = shap.TreeExplainer(model_shap)
            shap_vals = explainer.shap_values(X_shap)

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_vals, X_shap, feature_names=cols_shap,
                              show=False, max_display=20)
            plt.title(f'SHAP Summary — {best_tree}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{PLOT_DIR}/28_best_shap_v2.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  Saved 28_best_shap_v2.png")
        except Exception as e:
            print(f"  SHAP failed: {e}")

# ============================================================
# 13. SUMMARY
# ============================================================
elapsed = time.time() - t0_global
print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE ({elapsed:.0f}s)")
print(f"{'='*70}")

best = res_df.iloc[0]
print(f"Best model by LOO R²: {best['Model']} — LOO R²={best['LOO_R2']:.4f}")

# Best by BIC among well-performing models
bic_valid = res_df[(res_df['BIC'].notna()) & (res_df['LOO_R2'] > 0.5)].sort_values('BIC')
if len(bic_valid) > 0:
    best_bic = bic_valid.iloc[0]
    print(f"Best model by BIC (among R²>0.5): {best_bic['Model']} — "
          f"BIC={best_bic['BIC']:.1f}, LOO R²={best_bic['LOO_R2']:.4f}, k_eff={best_bic['k_eff']:.0f}")

print(f"\nOptimal grain-size exponent: n = {n_opt:.3f}")
print(f"(cf. Hall-Petch n=0.500, FCC average n≈0.40 per Cordero et al. 2016)")
