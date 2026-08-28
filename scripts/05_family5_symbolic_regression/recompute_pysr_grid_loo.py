#!/usr/bin/env python3
"""Recompute LOO R² for every cell in pysr_grid_summary_{YS,HV}.csv.

The Colab notebook averaged per-fold R² across LOO folds; with n_test=1
per fold, sklearn.r2_score returns NaN (variance of a single value is 0).
The correct LOO R² is computed by collecting predictions across all n
folds and computing one global r2_score on the aggregated vectors.

This script reads each cell's pareto.csv, refits the accuracy- and
elbow-selected equations under LOO via scipy.optimize.least_squares
(same machinery as the notebook), aggregates predictions globally, and
patches the summary CSVs.
"""
from __future__ import annotations
import warnings
import re
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR
warnings.filterwarnings('ignore')

GRID_DIR = Path(RESULTS_DIR) / 'pysr_grid'

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
FEATURE_SETS = {
    'F1_grain': ['d_inv_sqrt', 'SD_GS'],
    'F2_full':  [f'{e}_frac' for e in ELEMENTS] + ['ColdWork','RecrystT','HoldTime','d_inv_sqrt','SD_GS'],
    'F3_wen':   ['VEC','dH_mix','dS_mix','Omega','delta_chi','delta','ColdWork','RecrystT','HoldTime','d_inv_sqrt','SD_GS'],
}


def replace_floats_with_symbols(expr):
    syms, inits, counter = [], [], [0]
    def walk(e):
        if isinstance(e, sp.Float):
            s = sp.Symbol(f'c_{counter[0]}', real=True)
            counter[0] += 1; syms.append(s); inits.append(float(e))
            return s
        if e.is_Atom: return e
        return e.func(*[walk(a) for a in e.args])
    return walk(expr), syms, inits


def find_elbow_idx(complexity, loss):
    if len(complexity) < 3:
        return len(complexity) - 1
    loss_log = np.log(np.clip(loss, 1e-30, None))
    c = (complexity - complexity.min()) / max(complexity.max() - complexity.min(), 1e-12)
    l = (loss_log - loss_log.min()) / max(loss_log.max() - loss_log.min(), 1e-12)
    p1, p2 = np.array([c[0], l[0]]), np.array([c[-1], l[-1]])
    v = p2 - p1
    v_norm = v / (np.linalg.norm(v) + 1e-12)
    dists = np.zeros(len(c))
    for i in range(len(c)):
        p = np.array([c[i], l[i]])
        proj = p1 + np.dot(p - p1, v_norm) * v_norm
        dists[i] = np.linalg.norm(p - proj)
    return int(np.argmax(dists))


def loo_r2_global(eq_str, feature_cols, X, y):
    """Refit constants of `eq_str` under LOO. Aggregate predictions across
    all n folds, then compute one global r2_score. Returns (r2, rmse,
    n_constants)."""
    try:
        # PySR's O3 operator set introduces unary operators that sympy
        # doesn't recognize by name. Translate `square(x)` → x**2 and
        # `cube(x)` → x**3 before parsing. (sqrt and log are sympy-native.)
        local_dict = {c: sp.Symbol(c) for c in feature_cols}
        local_dict['square'] = lambda x: x**2
        local_dict['cube']   = lambda x: x**3
        expr = sp.parse_expr(eq_str, local_dict=local_dict)
    except Exception as exc:
        return np.nan, np.nan, 0, f'parse_failed: {exc}'

    param_expr, param_syms, init_vals = replace_floats_with_symbols(expr)
    feat_syms = sp.symbols(feature_cols)
    all_syms = list(feat_syms) + list(param_syms)
    f = sp.lambdify(all_syms, param_expr, modules=['numpy'])

    def predict(X_, params):
        args = [X_[:, i] for i in range(X_.shape[1])]
        out = f(*args, *params) if len(params) else f(*args)
        out = np.asarray(out, dtype=float)
        return np.full(X_.shape[0], float(out)) if out.ndim == 0 else out

    def residuals(params, X_, y_):
        return predict(X_, params) - y_

    n = len(y)
    preds = np.full(n, np.nan)
    for tr, te in LeaveOneOut().split(X):
        Xtr, ytr = X[tr], y[tr]
        Xte = X[te]
        if not init_vals:
            preds[te] = predict(Xte, [])
            continue
        try:
            res = least_squares(residuals, init_vals,
                                args=(Xtr, ytr), max_nfev=5000)
            yp = predict(Xte, res.x)
        except Exception:
            yp = predict(Xte, init_vals)
        if not np.all(np.isfinite(yp)):
            yp = np.where(np.isfinite(yp), yp, ytr.mean())
        preds[te] = yp
    ok = np.isfinite(preds)
    if ok.sum() < n / 2:
        return np.nan, np.nan, len(init_vals), 'too_many_failed_folds'
    r2 = float(r2_score(y[ok], preds[ok]))
    rmse = float(np.sqrt(mean_squared_error(y[ok], preds[ok])))
    return r2, rmse, len(init_vals), 'ok'


def main():
    df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
    print(f'Loaded {len(df)} alloys')

    for tgt in ['YS', 'HV']:
        summary_path = Path(RESULTS_DIR) / f'pysr_grid_summary_{tgt}.csv'
        if not summary_path.exists():
            print(f'[skip] {summary_path} missing'); continue
        summary = pd.read_csv(summary_path)
        print(f'\n=== {tgt}: {len(summary)} rows in {summary_path.name} ===')

        # For each unique cell (feature_set × op_set), recompute LOO for
        # both accuracy and elbow selections by re-reading the pareto CSV
        for (fs, op), grp in summary.groupby(['feature_set', 'op_set']):
            pareto_path = GRID_DIR / f'{tgt}__{fs}__{op}_pareto.csv'
            if not pareto_path.exists():
                print(f'  [skip] {pareto_path.name} missing'); continue
            pareto = pd.read_csv(pareto_path)
            feats = [c for c in FEATURE_SETS[fs] if c in df.columns]
            sub = df.dropna(subset=feats + [tgt]).reset_index(drop=True)
            X = sub[feats].values.astype(float)
            y = sub[tgt].values.astype(float)
            for sel_type in ('accuracy', 'elbow'):
                if sel_type == 'accuracy':
                    idx = int(pareto['loss'].idxmin())
                else:
                    idx = find_elbow_idx(pareto['complexity'].values, pareto['loss'].values)
                eq = str(pareto.iloc[idx]['equation'])
                r2, rmse, nc, status = loo_r2_global(eq, feats, X, y)
                mask = ((summary['feature_set'] == fs) &
                        (summary['op_set'] == op) &
                        (summary['selection'] == sel_type))
                summary.loc[mask, 'cv_refit_LOO_R2'] = round(r2, 3) if not np.isnan(r2) else np.nan
                print(f'  {tgt}/{fs}/{op}/{sel_type:8s}: LOO_R²={r2:.3f}, RMSE={rmse:.1f} ({status})')

        summary.to_csv(summary_path, index=False)
        print(f'  Wrote {summary_path}')


if __name__ == '__main__':
    main()
