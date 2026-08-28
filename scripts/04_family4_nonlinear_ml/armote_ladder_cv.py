#!/usr/bin/env python3
"""
ARMOTE on the S1-S4 ladder, reporting ALL THREE CV protocols (fair comparison).
===============================================================================

Runs the ARMOTE non-linear panel (with Optuna HPO) on the revised S1-S4 feature
ladder read from data/derived/inputs.csv, and reports 5-fold, LOO, and LOBO R^2
for every model x feature-set x target (YS and HV). This closes the gap where
the canonical ARMOTE table (model_search_results_v2.csv) reported LOO+LOBO but
dropped 5-fold, so ARMOTE can be compared on equal footing with the linear
models (fair_comparison.csv) and the symbolic-regression tier.

CV protocols (all reported, same seeds as the rest of the repo):
  - 5-fold : KFold(n_splits=5, shuffle, seed 42)
  - LOO    : LeaveOneOut
  - LOBO   : LeaveOneGroupOut on Iteration (BO batch / cluster)

HPO: Optuna TPE, 25 trials, single 5-fold objective, 90 s/model cap. The final
5-fold/LOO/LOBO are recomputed with the tuned params (deterministic). Linear
models also get a BIC (k = n_features + 1).

Output: results/armote_ladder_cv.csv
"""
import warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import (LeaveOneOut, LeaveOneGroupOut, KFold,
                                     cross_val_score)
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR

SEED = 42
ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# ---- load inputs.csv + manifest ------------------------------------------
df = pd.read_csv(f'{DATA_DIR}/inputs.csv')
man = pd.read_csv(f'{DATA_DIR}/inputs_feature_manifest.csv')
blocks = {b: man[man.block == b]['column'].tolist() for b in man.block.unique()}
S1 = blocks['S1_grain']
S2 = blocks['S2_wen']
S3 = blocks['S3_proc']
S4 = blocks['S4_comp'] + blocks['S4_sss']

# cumulative ladder S1 -> S1-4 (matches fair_comparison tiers)
LADDER = {
    'S1_grain':    S1,
    'S2_wen':      S1 + S2,
    'S3_wen_proc': S1 + S2 + S3,
    'S4_phys':     S1 + S2 + S3 + S4,
}

# ---- CV evaluators (all three protocols) ---------------------------------
def _oof(factory, X, y, splitter, groups=None):
    pred = np.full(len(y), np.nan)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X)
    for tr, te in it:
        m = factory(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred

def eval_all(factory, X, y, groups):
    p5 = _oof(factory, X, y, KFold(n_splits=5, shuffle=True, random_state=SEED))
    pl = _oof(factory, X, y, LeaveOneOut())
    pb = _oof(factory, X, y, LeaveOneGroupOut(), groups)
    return {
        'R2_5fold': r2_score(y, p5), 'RMSE_5fold': np.sqrt(mean_squared_error(y, p5)),
        'LOO_R2':   r2_score(y, pl), 'LOO_RMSE':   np.sqrt(mean_squared_error(y, pl)),
        'LOBO_R2':  r2_score(y, pb), 'LOBO_RMSE':  np.sqrt(mean_squared_error(y, pb)),
    }

# ---- Optuna spaces (subset of exhaustive_model_search.py) ----------------
def ridge_space(t): return {'alpha': t.suggest_float('alpha', 1e-3, 100, log=True)}
def enet_space(t):  return {'alpha': t.suggest_float('alpha', 1e-4, 10, log=True),
                            'l1_ratio': t.suggest_float('l1_ratio', 0.01, 0.99), 'max_iter': 10000}
def svr_space(t):   return {'C': t.suggest_float('C', 0.1, 1000, log=True),
                            'epsilon': t.suggest_float('epsilon', 0.01, 10, log=True),
                            'gamma': t.suggest_categorical('gamma', ['scale', 'auto'])}
def krr_space(t):   return {'alpha': t.suggest_float('alpha', 1e-3, 100, log=True),
                            'kernel': 'rbf', 'gamma': t.suggest_float('gamma', 1e-4, 1.0, log=True)}
def rf_space(t):    return {'n_estimators': t.suggest_int('n_estimators', 100, 500),
                            'max_depth': t.suggest_int('max_depth', 3, 8),
                            'min_samples_leaf': t.suggest_int('min_samples_leaf', 3, 15),
                            'max_features': t.suggest_float('max_features', 0.3, 1.0), 'random_state': SEED}
def xgb_space(t):   return {'n_estimators': t.suggest_int('n_estimators', 50, 700),
                            'max_depth': t.suggest_int('max_depth', 2, 5),
                            'learning_rate': t.suggest_float('learning_rate', 0.005, 0.2, log=True),
                            'subsample': t.suggest_float('subsample', 0.5, 1.0),
                            'colsample_bytree': t.suggest_float('colsample_bytree', 0.3, 1.0),
                            'min_child_weight': t.suggest_int('min_child_weight', 3, 15),
                            'reg_lambda': t.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                            'objective': 'reg:squarederror', 'random_state': SEED, 'verbosity': 0}
def lgbm_space(t):  return {'n_estimators': t.suggest_int('n_estimators', 50, 700),
                            'max_depth': t.suggest_int('max_depth', 2, 6),
                            'learning_rate': t.suggest_float('learning_rate', 0.005, 0.2, log=True),
                            'num_leaves': t.suggest_int('num_leaves', 7, 63),
                            'min_child_samples': t.suggest_int('min_child_samples', 5, 30),
                            'subsample': t.suggest_float('subsample', 0.5, 1.0),
                            'colsample_bytree': t.suggest_float('colsample_bytree', 0.3, 1.0),
                            'random_state': SEED, 'verbosity': -1}

# model -> (class, space, needs_scaling)
PANEL = {
    'Ridge':        (Ridge, ridge_space, True),
    'ElasticNet':   (ElasticNet, enet_space, True),
    'SVR (RBF)':    (SVR, svr_space, True),
    'KRR (RBF)':    (KernelRidge, krr_space, True),
    'Random Forest':(RandomForestRegressor, rf_space, False),
    'XGBoost':      (xgb.XGBRegressor, xgb_space, False),
    'LightGBM':     (lgb.LGBMRegressor, lgbm_space, False),
}

def make_factory(cls, params, scale):
    if scale:
        return lambda: Pipeline([('s', StandardScaler()), ('m', cls(**params))])
    return lambda: cls(**params)

def run_optuna(cls, space, X, y, scale, n_trials=25, timeout=90):
    def obj(trial):
        params = space(trial)
        est = Pipeline([('s', StandardScaler()), ('m', cls(**params))]) if scale else cls(**params)
        return cross_val_score(est, X, y, cv=KFold(5, shuffle=True, random_state=SEED),
                               scoring='neg_mean_squared_error').mean()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params

def bic_linear(y, pred, k):
    n = len(y); rss = max(np.sum((y - pred) ** 2), 1e-12)
    return n * np.log(rss / n) + k * np.log(n)

# ---- main loop -----------------------------------------------------------
rows = []
for target in ['YS', 'HV']:
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    y = sub[target].values.astype(float)
    groups = sub['Iteration'].values
    print(f'\n{"="*70}\nARMOTE ladder CV — {target} (n={len(y)})\n{"="*70}')
    for fs_name, cols in LADDER.items():
        X = sub[cols].fillna(0.0).values.astype(float)
        # OLS baseline (no HPO)
        for name, factory, lin_k in [
            ('OLS', lambda: Pipeline([('s', StandardScaler()), ('m', LinearRegression())]), X.shape[1] + 1)]:
            m = eval_all(factory, X, y, groups)
            pl = _oof(factory, X, y, LeaveOneOut())
            rows.append({'Target': target, 'FeatureSet': fs_name, 'Model': name, 'n_feat': X.shape[1],
                         **{k: round(v, 4) for k, v in m.items()},
                         'BIC': round(bic_linear(y, pl, lin_k), 1), 'HPO': 'none'})
            print(f'  [{fs_name:11s}] {name:13s} 5f={m["R2_5fold"]:.3f} LOO={m["LOO_R2"]:.3f} LOBO={m["LOBO_R2"]:.3f}')
        # HPO panel
        for name, (cls, space, scale) in PANEL.items():
            t0 = time.time()
            try:
                best = run_optuna(cls, space, X, y, scale)
                factory = make_factory(cls, best, scale)
                m = eval_all(factory, X, y, groups)
                bic = ''
                if name in ('Ridge', 'ElasticNet'):
                    pl = _oof(factory, X, y, LeaveOneOut())
                    bic = round(bic_linear(y, pl, X.shape[1] + 1), 1)
                rows.append({'Target': target, 'FeatureSet': fs_name, 'Model': name, 'n_feat': X.shape[1],
                             **{k: round(v, 4) for k, v in m.items()}, 'BIC': bic, 'HPO': 'optuna-25'})
                print(f'  [{fs_name:11s}] {name:13s} 5f={m["R2_5fold"]:.3f} LOO={m["LOO_R2"]:.3f} '
                      f'LOBO={m["LOBO_R2"]:.3f} ({time.time()-t0:.0f}s)')
            except Exception as e:
                print(f'  [{fs_name:11s}] {name:13s} FAILED: {e}')

res = pd.DataFrame(rows)
res.to_csv(f'{RESULTS_DIR}/armote_ladder_cv.csv', index=False)
print(f'\nWrote results/armote_ladder_cv.csv ({len(res)} rows)')

# ---- summary: best ARMOTE model per tier, all three CV -------------------
for target in ['YS', 'HV']:
    print(f'\n{"="*70}\n{target}: best ARMOTE model per tier (by LOBO)\n{"="*70}')
    t = res[res.Target == target]
    for fs in LADDER:
        s = t[t.FeatureSet == fs].sort_values('LOBO_R2', ascending=False)
        if len(s):
            b = s.iloc[0]
            print(f'  {fs:11s} {b.Model:13s} 5f={b.R2_5fold:+.3f} LOO={b.LOO_R2:+.3f} LOBO={b.LOBO_R2:+.3f}')
print('\nDone.')
