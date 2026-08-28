#!/usr/bin/env python3
"""
Fair linear-vs-non-linear comparison for FCC-HEA YS and HV.
===========================================================

The headline tables in the paper are NOT apples-to-apples: each model was
assigned its own feature set (Ridge -> PHYSICS-33, XGBoost -> INTERACTIONS-64,
PCA-OLS -> curated Wen + SD_grain), and the tree models were tuned with 50
Optuna trials while the linear models ran at defaults. This script removes
every confound *except the hypothesis class* so that "linear vs ARMOTE
non-linear" is a fair fight.

Fairness controls (the whole point):
  1. SAME feature matrix X. A fixed LADDER of feature sets is defined once and
     every model is run on every set. The curated Wen descriptors and SD_grain
     that the linear models use are now ALSO given to the non-linear models,
     and the engineered interaction columns the trees used are ALSO given to
     the linear models.
  2. SAME rows + SAME CV splits. Identical dropna, identical LOO, identical
     LOBO (leave-one-batch-out on Iteration), identical 5-fold (seed 42).
  3. SAME tuning budget. All models run at fixed, sensible defaults -> equal
     (zero) tuning budget. No model gets an HPO advantage.
  4. Complexity-aware. n_features reported for every row; BIC reported for the
     linear family where effective parameters are well defined.
  5. SAME generalization emphasis: 5-fold, LOO, and LOBO reported for all,
     plus the LOO - LOBO gap (the overfitting tax).

Feature ladder (cumulative):
  S1_grain      : d^-1/2, SD_GS                                  (grain geometry only)
  S2_wen        : S1 + curated Wen [VEC,dH_mix,dS_mix,Omega,delta_chi,delta]
  S3_wen_proc   : S2 + processing [ColdWork,RecrystT,HoldTime]
  S4_phys       : S3 + composition fractions + SSS sigmas (VLC,Labusch,TC) + mu_bar,delta_mu,Tm_bar,a_bar
  S5_interact   : S4 + composition x d^-1/2 + SSS x d^-1/2 + proc x d^-1/2 + quadratics

Outputs:
  results/fair_comparison.csv         (long form, every Target x FeatureSet x Model row)
  results/fair_comparison_YS.csv
  results/fair_comparison_HV.csv
  analysis_plots/fair_comparison_LOBO_heatmap.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
SEED = 42

# ============================================================
# 1. LOAD + ENGINEER (identical feature engineering for everyone)
# ============================================================
print('=' * 72)
print('FAIR LINEAR vs NON-LINEAR COMPARISON (YS + HV)')
print('=' * 72)

df = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
df = df.loc[:, ~df.columns.duplicated()].copy()
df['Omega'] = df['Omega'].clip(upper=100)
df = df.replace([np.inf, -np.inf], np.nan)

# Engineered interactions — given to ALL families (fairness control #1b)
for el in ELEMENTS:
    df[f'{el}_x_dinv'] = df[f'{el}_frac'] * df['d_inv_sqrt']
df['VLC_x_dinv']      = df['sigma_y0_VLC'] * df['d_inv_sqrt']
df['Labusch_x_dinv']  = df['sigma_Labusch'] * df['d_inv_sqrt']
df['TC_x_dinv']       = df['sigma_TC'] * df['d_inv_sqrt']
df['RecrystT_x_dinv'] = df['RecrystT'] * df['d_inv_sqrt']
df['CW_x_dinv']       = df['ColdWork'] * df['d_inv_sqrt']
df['SD_x_dinv']       = df['SD_GS'] * df['d_inv_sqrt']
df['dinv_sq']         = df['d_inv_sqrt'] ** 2
df['V_frac_sq']       = df['V_frac'] ** 2

CURATED_WEN = ['VEC', 'dH_mix', 'dS_mix', 'Omega', 'delta_chi', 'delta']
GRAIN  = ['d_inv_sqrt', 'SD_GS']
PROC   = ['ColdWork', 'RecrystT', 'HoldTime']
COMPF  = [f'{el}_frac' for el in ELEMENTS]
SSS    = ['sigma_y0_VLC', 'sigma_Labusch', 'sigma_TC', 'Phi_VLC', 'eps_Labusch']
PHYS_X = ['mu_bar', 'delta_mu', 'Tm_bar', 'a_bar']
INTER  = [f'{el}_x_dinv' for el in ELEMENTS] + \
         ['VLC_x_dinv', 'Labusch_x_dinv', 'TC_x_dinv',
          'RecrystT_x_dinv', 'CW_x_dinv', 'SD_x_dinv', 'dinv_sq', 'V_frac_sq']

# S4 uses composition fractions + SSS descriptors only; the rule-of-mixtures
# MEAN-physics terms (mu_bar, delta_mu, Tm_bar, a_bar) were dropped because they
# never survive symbolic-model selection (only delta_mu appears, as the SISSO
# Full singularity) and are crowded out by variance/misfit descriptors.
# S5 (hand-engineered interactions) is RETIRED: nonlinear feature discovery is
# now handled by the PySR symbolic-regression grid (scripts/pysr_grid_analysis.py),
# the principled "let the search find the interactions" tier.
FEATURE_LADDER = {
    'S1_grain'    : GRAIN,
    'S2_wen'      : GRAIN + CURATED_WEN,
    'S3_wen_proc' : GRAIN + CURATED_WEN + PROC,
    'S4_phys'     : GRAIN + CURATED_WEN + PROC + COMPF + SSS,
}
for k, v in FEATURE_LADDER.items():
    miss = [c for c in v if c not in df.columns]
    assert not miss, f'{k} missing {miss}'
    print(f'  {k:13s} : {len(v)} features')

# ============================================================
# 2. MODELS — equal (zero) tuning budget, fixed sensible defaults
# ============================================================
def make_models():
    """Fresh estimator dict each call (avoids state leakage across folds)."""
    return {
        # ---- LINEAR family ----
        ('linear', 'OLS')        : Pipeline([('sc', StandardScaler()), ('m', LinearRegression())]),
        ('linear', 'Ridge')      : Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0, random_state=SEED))]),
        ('linear', 'Lasso')      : Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=1.0, random_state=SEED, max_iter=10000))]),
        ('linear', 'ElasticNet') : Pipeline([('sc', StandardScaler()), ('m', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=SEED, max_iter=10000))]),
        # ---- NON-LINEAR family (ARMOTE panel) ----
        ('nonlinear', 'SVR-RBF')      : Pipeline([('sc', StandardScaler()), ('m', SVR(kernel='rbf', C=10.0, gamma='scale'))]),
        ('nonlinear', 'KRR-RBF')      : Pipeline([('sc', StandardScaler()), ('m', KernelRidge(kernel='rbf', alpha=1.0, gamma=None))]),
        ('nonlinear', 'RandomForest') : RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        ('nonlinear', 'ExtraTrees')   : ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        ('nonlinear', 'GradBoost')    : GradientBoostingRegressor(random_state=SEED),
        ('nonlinear', 'XGBoost')      : XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                                     subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0),
        ('nonlinear', 'LightGBM')     : LGBMRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                                      subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1),
        ('nonlinear', 'MLP')          : Pipeline([('sc', StandardScaler()),
                                                  ('m', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                                                     early_stopping=True, random_state=SEED))]),
    }


def pca_ols(n_pc=6):
    """Linear PCA-OLS, mirrors scripts/pca_ols_analysis.py."""
    return Pipeline([('sc', StandardScaler()),
                     ('pca', PCA(n_components=n_pc, random_state=SEED)),
                     ('m', LinearRegression())])


def cv_predict(model_factory, X, y, splitter, groups=None):
    """Out-of-fold predictions for an arbitrary splitter, refit per fold."""
    pred = np.full(len(y), np.nan)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X)
    for tr, te in it:
        m = model_factory()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def score(y, p):
    return r2_score(y, p), float(np.sqrt(mean_squared_error(y, p)))


def bic_linear(y, p, k):
    """BIC for a Gaussian-noise linear model with k effective parameters."""
    n = len(y)
    rss = float(np.sum((y - p) ** 2))
    return n * np.log(rss / n) + k * np.log(n)


# ============================================================
# 3. RUN — every Target x FeatureSet x Model on identical splits
# ============================================================
def run_target(target):
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    y = sub[target].values.astype(float)
    groups = sub['Iteration'].values
    loo  = LeaveOneOut()
    loco = LeaveOneGroupOut()
    kf   = KFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f'\n{"="*72}\nTARGET = {target}  (n={len(y)}, clusters={len(set(groups))})\n{"="*72}')

    rows = []
    for fs_name, cols in FEATURE_LADDER.items():
        X = sub[cols].fillna(0.0).values.astype(float)
        # assemble model list incl. PCA-OLS (only meaningful when n_feat > n_pc)
        models = make_models()
        model_items = list(models.items())
        if X.shape[1] >= 6:
            model_items.append((('linear', 'PCA-OLS(6)'), 'PCAOLS'))

        for (family, name), est in model_items:
            factory = (lambda e=est: pca_ols(6)) if est == 'PCAOLS' else (lambda e=est: __import__('sklearn').base.clone(e))
            try:
                p_loo  = cv_predict(factory, X, y, loo)
                p_loco = cv_predict(factory, X, y, loco, groups=groups)
                p_5f   = cv_predict(factory, X, y, kf)
                r2_loo,  rmse_loo  = score(y, p_loo)
                r2_loco, rmse_loco = score(y, p_loco)
                r2_5f,   rmse_5f   = score(y, p_5f)
                kf_eff = (6 if name == 'PCA-OLS(6)' else X.shape[1]) + 1
                bic = bic_linear(y, p_loo, kf_eff) if family == 'linear' else np.nan
                rows.append({
                    'Target': target, 'FeatureSet': fs_name, 'n_feat': X.shape[1],
                    'Family': family, 'Model': name,
                    'R2_5fold': round(r2_5f, 4), 'R2_LOO': round(r2_loo, 4), 'R2_LOBO': round(r2_loco, 4),
                    'RMSE_LOO': round(rmse_loo, 2), 'RMSE_LOBO': round(rmse_loco, 2),
                    'LOO_minus_LOBO': round(r2_loo - r2_loco, 4),
                    'BIC_linear': round(bic, 1) if np.isfinite(bic) else '',
                })
                print(f'  [{fs_name:11s}] {family:9s} {name:13s}  '
                      f'5f={r2_5f:.3f}  LOO={r2_loo:.3f}  LOBO={r2_loco:.3f}  gap={r2_loo-r2_loco:+.3f}')
            except Exception as e:
                print(f'  [{fs_name:11s}] {family:9s} {name:13s}  FAILED: {e}')
    return pd.DataFrame(rows)


res_ys = run_target('YS')
res_hv = run_target('HV')
res_all = pd.concat([res_ys, res_hv], ignore_index=True)

res_ys.to_csv(f'{RESULTS_DIR}/fair_comparison_YS.csv', index=False)
res_hv.to_csv(f'{RESULTS_DIR}/fair_comparison_HV.csv', index=False)
res_all.to_csv(f'{RESULTS_DIR}/fair_comparison.csv', index=False)
print(f'\nWrote results/fair_comparison{{,_YS,_HV}}.csv')

# ============================================================
# 4. PLOT — LOBO R^2 heatmap (model x feature set), faceted by target
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (tgt, rdf) in zip(axes, [('YS', res_ys), ('HV', res_hv)]):
    piv = rdf.pivot_table(index='Model', columns='FeatureSet', values='R2_LOBO')
    piv = piv.reindex(columns=list(FEATURE_LADDER.keys()))
    # order rows: linear first then nonlinear, by best LOBO
    order = (rdf.groupby('Model')['R2_LOBO'].max().sort_values(ascending=False).index.tolist())
    piv = piv.reindex(order)
    im = ax.imshow(piv.values, cmap='RdYlGn', vmin=0, vmax=0.8, aspect='auto')
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=10)
    fam_map = rdf.set_index('Model')['Family'].to_dict()
    for i, m in enumerate(piv.index):
        ax.get_yticklabels()[i].set_color('#1F3A5F' if fam_map.get(m) == 'linear' else '#C84B31')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='black' if v > 0.3 else 'white')
    ax.set_title(f'{tgt}: LOBO $R^2$ (blue=linear, red=non-linear)', fontsize=12, weight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='LOBO $R^2$')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/fair_comparison_LOBO_heatmap.png', dpi=150, bbox_inches='tight')
print(f'Wrote analysis_plots/fair_comparison_LOBO_heatmap.png')

# ============================================================
# 5. SUMMARY — best linear vs best non-linear per (target, feature set)
# ============================================================
print(f'\n{"="*72}\nBEST LINEAR vs BEST NON-LINEAR by LOBO R^2\n{"="*72}')
for tgt, rdf in [('YS', res_ys), ('HV', res_hv)]:
    print(f'\n{tgt}:')
    for fs in FEATURE_LADDER:
        s = rdf[rdf.FeatureSet == fs]
        lin = s[s.Family == 'linear'].sort_values('R2_LOBO', ascending=False).head(1)
        nl  = s[s.Family == 'nonlinear'].sort_values('R2_LOBO', ascending=False).head(1)
        if len(lin) and len(nl):
            ln, lr = lin.iloc[0]['Model'], lin.iloc[0]['R2_LOBO']
            nn, nr = nl.iloc[0]['Model'], nl.iloc[0]['R2_LOBO']
            win = 'LINEAR' if lr >= nr else 'NON-LIN'
            print(f'  {fs:13s}  linear={ln:12s}({lr:+.3f})  nonlin={nn:12s}({nr:+.3f})  -> {win} by {abs(lr-nr):.3f}')
print('\nDone.')
