#!/usr/bin/env python3
"""PySR grid: 2 targets x 3 feature sets x 3 operator sets = 18 cells.

Ported from the proven-working sr_runner.py in the parent "symbolic regression"
folder. Critical difference from the previous (hanging) implementation:

  1. Call PySR ONCE per cell with model_selection="accuracy" to retrieve the
     full Pareto front.
  2. Do NOT re-fit PySR per CV fold. Instead, extract the sympy expression
     for the chosen equation, replace every numeric constant with a unique
     sympy.Symbol, lambdify, and refit constants in each fold via
     scipy.optimize.least_squares.
  3. Two CV variants per chosen equation (accuracy-selected and elbow-
     selected): "refit" (refit constants per fold) and "frozen" (apply the
     full-data equation as-is to each held-out fold).
  4. Three CV protocols reported side by side: 5-fold KFold, leave-one-out,
     and leave-one-batch-out.

This bypasses the PySR multi-fit hang that the previous implementation hit:
PySR runs only once per cell, and the scipy-refit machinery is what feeds
all three CV protocols.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

# ----------------- config (mirrors sr_runner.py knobs) ----------------------
SEED = 42
N_FOLDS = 5

# Feature sets — F1 grain-only, F2 raw composition, F3 Wen-curated.
# The original sr_runner uses a separate Wen file; here we use the
# data_with_descriptors.csv columns produced by eda_analysis.py.
ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
F1_FEATURES = ['d_inv_sqrt', 'SD_GS']
F2_FEATURES = [f'{e}_frac' for e in ELEMENTS] + ['ColdWork', 'RecrystT',
                                                   'HoldTime',
                                                   'd_inv_sqrt', 'SD_GS']
F3_FEATURES = ['VEC', 'dH_mix', 'dS_mix', 'Omega', 'delta_chi', 'delta',
               'ColdWork', 'RecrystT', 'HoldTime',
               'd_inv_sqrt', 'SD_GS']

FEATURE_SETS = {
    'F1_grain': (F1_FEATURES, 15),
    'F2_full':  (F2_FEATURES, 30),
    'F3_wen':   (F3_FEATURES, 30),
}

OPERATOR_SETS = {
    'O1_add_sub':         {'binary': ['+', '-'],
                           'unary':  [],
                           'niterations': 50},
    'O2_add_sub_mul_div': {'binary': ['+', '-', '*', '/'],
                           'unary':  [],
                           'niterations': 50},
    'O3_full':            {'binary': ['+', '-', '*', '/'],
                           'unary':  ['sqrt', 'log', 'square', 'cube'],
                           'niterations': 75},
}
# NOTE: niterations were reduced from the sr_runner.py originals (400/600)
# to 50/75 because pysr 1.5.10 + macOS 26.2 deadlocks on high iteration
# counts (>>100 iter hangs at 0% CPU). 50–75 iterations are sufficient at
# n=93 to reach a useful Pareto front; this is documented in the report
# as a constrained-search caveat.

PYSR_GRID_DIR = Path(RESULTS_DIR) / 'pysr_grid'
PYSR_GRID_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- helpers (ported from sr_runner.py) -----------------------

def build_pysr(binary_ops, unary_ops, maxsize, niterations):
    """Exact replication of sr_runner.build_pysr — proven to run 18 cells
    sequentially without hanging."""
    kw = dict(
        niterations=niterations,
        binary_operators=binary_ops,
        maxsize=maxsize,
        model_selection='accuracy',
        random_state=SEED,
        deterministic=True,
        parallelism='serial',
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    if unary_ops:
        kw['unary_operators'] = unary_ops
    return PySRRegressor(**kw)


def find_elbow_idx(complexity: np.ndarray, loss: np.ndarray) -> int:
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


def replace_floats_with_symbols(expr):
    syms, inits = [], []
    counter = [0]

    def walk(e):
        if isinstance(e, sp.Float):
            s = sp.Symbol(f'c_{counter[0]}', real=True)
            counter[0] += 1
            syms.append(s)
            inits.append(float(e))
            return s
        if e.is_Atom:
            return e
        return e.func(*[walk(a) for a in e.args])

    return walk(expr), syms, inits


def posthoc_cv(model, eq_idx, X, y, feature_cols, splitter, groups=None,
               refit=True):
    """Generalized post-hoc CV. If refit=True, refit constants per fold via
    scipy.optimize.least_squares; if False, predict with the full-data
    equation as-is."""
    expr = model.sympy(index=eq_idx)
    param_expr, param_syms, init_vals = replace_floats_with_symbols(expr)
    feat_syms = sp.symbols(feature_cols)
    all_syms = list(feat_syms) + list(param_syms)
    f = sp.lambdify(all_syms, param_expr, modules=['numpy'])

    def predict(X_, params):
        feat_args = [X_[:, i] for i in range(X_.shape[1])]
        out = f(*feat_args, *params) if len(params) else f(*feat_args)
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:
            out = np.full(X_.shape[0], float(out))
        return out

    def residuals(params, X_, y_):
        return predict(X_, params) - y_

    iter_args = (X, y, groups) if groups is not None else (X,)
    r2_per_fold, mse_per_fold = [], []
    for tr, te in splitter.split(*iter_args):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        if not refit or len(init_vals) == 0:
            y_pred = predict(X_te, init_vals)
        else:
            try:
                res = least_squares(residuals, init_vals,
                                    args=(X_tr, y_tr), max_nfev=5000)
                y_pred = predict(X_te, res.x)
            except Exception:
                y_pred = predict(X_te, init_vals)
        if not np.all(np.isfinite(y_pred)):
            y_pred = np.where(np.isfinite(y_pred), y_pred, y_tr.mean())
        r2_per_fold.append(float(r2_score(y_te, y_pred)))
        mse_per_fold.append(float(mean_squared_error(y_te, y_pred)))
    return {
        'cv_r2': float(np.mean(r2_per_fold)),
        'cv_mse': float(np.mean(mse_per_fold)),
        'cv_r2_folds': r2_per_fold,
        'cv_mse_folds': mse_per_fold,
        'n_constants': len(init_vals),
        'parameterized_expression': str(param_expr),
    }


def all_three_cvs(model, eq_idx, X, y, feature_cols, groups, refit=True):
    """Run 5-fold + LOO + LOBO via the same post-hoc-CV machinery."""
    out = {}
    out['5fold'] = posthoc_cv(model, eq_idx, X, y, feature_cols,
                              KFold(n_splits=N_FOLDS, shuffle=True,
                                    random_state=SEED), refit=refit)
    out['LOO'] = posthoc_cv(model, eq_idx, X, y, feature_cols,
                            LeaveOneOut(), refit=refit)
    if groups is not None:
        out['LOBO'] = posthoc_cv(model, eq_idx, X, y, feature_cols,
                                 LeaveOneGroupOut(), groups=groups,
                                 refit=refit)
    else:
        out['LOBO'] = None
    return out


# ----------------- experiment runner ----------------------------------------

def run_one_cell(df, target_col, fs_name, feature_cols, maxsize,
                 op_name, op_config, groups):
    run_tag = f'{target_col}__{fs_name}__{op_name}'
    print(f'\n[{run_tag}] n_features={len(feature_cols)} maxsize={maxsize} '
          f'niter={op_config["niterations"]}')

    feature_cols = [c for c in feature_cols if c in df.columns]
    sub = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    X = sub[feature_cols].values.astype(float)
    y = sub[target_col].values.astype(float)
    grp = sub['Iteration'].values if 'Iteration' in sub.columns else None
    print(f'  complete cases: {len(y)}; batches: {sorted(set(grp)) if grp is not None else "N/A"}')

    model = build_pysr(op_config['binary'], op_config['unary'],
                       maxsize, op_config['niterations'])
    print('  fitting PySR on full data...')
    model.fit(X, y, variable_names=feature_cols)
    pareto = model.equations_.copy().reset_index(drop=True)

    # Full-data R² per Pareto row
    full_r2, full_mse = [], []
    for i in range(len(pareto)):
        y_pred = model.predict(X, index=i)
        full_r2.append(r2_score(y, y_pred))
        full_mse.append(mean_squared_error(y, y_pred))
    pareto['full_r2'] = full_r2
    pareto['full_mse'] = full_mse
    pareto.to_csv(PYSR_GRID_DIR / f'{run_tag}_pareto.csv', index=False)

    idx_acc = int(pareto['loss'].idxmin())
    idx_elbow = find_elbow_idx(pareto['complexity'].values, pareto['loss'].values)
    print(f'  accuracy: complexity={pareto.loc[idx_acc, "complexity"]}, '
          f'full_R²={pareto.loc[idx_acc, "full_r2"]:.3f}')
    print(f'  elbow:    complexity={pareto.loc[idx_elbow, "complexity"]}, '
          f'full_R²={pareto.loc[idx_elbow, "full_r2"]:.3f}')

    # CV: refit and frozen for each selection
    cvs = {
        'accuracy_refit': all_three_cvs(model, idx_acc, X, y, feature_cols, grp, refit=True),
        'accuracy_frozen': all_three_cvs(model, idx_acc, X, y, feature_cols, grp, refit=False),
        'elbow_refit': all_three_cvs(model, idx_elbow, X, y, feature_cols, grp, refit=True),
        'elbow_frozen': all_three_cvs(model, idx_elbow, X, y, feature_cols, grp, refit=False),
    }
    for tag, cv in cvs.items():
        cv_str = ' '.join(
            f'{k}={v["cv_r2"]:.3f}' if v is not None else f'{k}=N/A'
            for k, v in cv.items()
        )
        print(f'  {tag:18s}  {cv_str}')

    # Plot Pareto
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pareto['complexity'], pareto['loss'], 'o-', color='steelblue',
            label='Pareto front')
    ax.plot(pareto.loc[idx_acc, 'complexity'], pareto.loc[idx_acc, 'loss'],
            's', color='crimson', markersize=12,
            label=f'Accuracy (c={pareto.loc[idx_acc, "complexity"]})')
    ax.plot(pareto.loc[idx_elbow, 'complexity'], pareto.loc[idx_elbow, 'loss'],
            'D', color='goldenrod', markersize=12,
            label=f'Elbow (c={pareto.loc[idx_elbow, "complexity"]})')
    ax.set_yscale('log')
    ax.set_xlabel('Complexity')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(f'Pareto: {run_tag}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PYSR_GRID_DIR / f'{run_tag}_pareto.png', dpi=120)
    plt.close(fig)

    return {
        'run_tag': run_tag,
        'target': target_col,
        'feature_set': fs_name,
        'op_set': op_name,
        'n_samples': int(len(y)),
        'accuracy': {
            'complexity': int(pareto.loc[idx_acc, 'complexity']),
            'equation': str(pareto.loc[idx_acc, 'equation']),
            'full_r2': float(pareto.loc[idx_acc, 'full_r2']),
            'cv': cvs['accuracy_refit'],
            'cv_frozen': cvs['accuracy_frozen'],
        },
        'elbow': {
            'complexity': int(pareto.loc[idx_elbow, 'complexity']),
            'equation': str(pareto.loc[idx_elbow, 'equation']),
            'full_r2': float(pareto.loc[idx_elbow, 'full_r2']),
            'cv': cvs['elbow_refit'],
            'cv_frozen': cvs['elbow_frozen'],
        },
    }


def build_summary_table(all_results):
    rows = []
    for s in all_results:
        for sel in ('accuracy', 'elbow'):
            d = s[sel]
            row = {
                'target': s['target'],
                'feature_set': s['feature_set'],
                'op_set': s['op_set'],
                'selection': sel,
                'complexity': d['complexity'],
                'n_constants': d['cv'].get('n_constants', np.nan),
                'fit_R2': round(d['full_r2'], 3),
                'cv_refit_5fold_R2':  round(d['cv']['5fold']['cv_r2'], 3),
                'cv_refit_LOO_R2':    round(d['cv']['LOO']['cv_r2'], 3),
                'cv_refit_LOBO_R2':   round(d['cv']['LOBO']['cv_r2'], 3) if d['cv']['LOBO'] else np.nan,
                'cv_frozen_5fold_R2': round(d['cv_frozen']['5fold']['cv_r2'], 3),
                'cv_frozen_LOO_R2':   round(d['cv_frozen']['LOO']['cv_r2'], 3),
                'cv_frozen_LOBO_R2':  round(d['cv_frozen']['LOBO']['cv_r2'], 3) if d['cv_frozen']['LOBO'] else np.nan,
                'equation': d['equation'],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_cell_jsons(target):
    """Re-read per-cell JSONs and rebuild the summary CSV + full JSON.
    Useful when cells were run as separate subprocesses."""
    cell_jsons = sorted(PYSR_GRID_DIR.glob(f'{target}__*__cell.json'))
    all_results = []
    for p in cell_jsons:
        with open(p) as fp:
            all_results.append(json.load(fp))
    if not all_results:
        print(f'[warn] no per-cell JSONs found for {target}')
        return
    summary_df = build_summary_table(all_results)
    out_csv = Path(RESULTS_DIR) / f'pysr_grid_summary_{target}.csv'
    summary_df.to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}  ({len(summary_df)} rows)')
    json_path = Path(RESULTS_DIR) / f'pysr_grid_full_{target}.json'
    with open(json_path, 'w') as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f'Wrote {json_path}')


def main():
    TARGET = os.environ.get('PYSR_TARGET', 'YS').upper()
    assert TARGET in ('YS', 'HV'), f'PYSR_TARGET={TARGET} must be YS or HV'

    # Aggregation-only mode (re-read per-cell JSONs and rebuild summary)
    if os.environ.get('PYSR_AGGREGATE_ONLY') == '1':
        _aggregate_cell_jsons(TARGET)
        return

    df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
    print(f'Loaded {len(df)} alloys; target = {TARGET}')

    # Single-cell mode (run only one cell, write per-cell JSON, exit)
    # — safest against multi-fit hangs because each cell gets a fresh process.
    single_fs = os.environ.get('PYSR_FS')      # e.g. 'F1_grain'
    single_op = os.environ.get('PYSR_OP')      # e.g. 'O1_add_sub'
    if single_fs and single_op:
        feature_cols, maxsize = FEATURE_SETS[single_fs]
        op_config = OPERATOR_SETS[single_op]
        s = run_one_cell(df, TARGET, single_fs, feature_cols, maxsize,
                         single_op, op_config, groups=None)
        cell_json = PYSR_GRID_DIR / f'{TARGET}__{single_fs}__{single_op}__cell.json'
        with open(cell_json, 'w') as fp:
            json.dump(s, fp, indent=2, default=str)
        print(f'Wrote {cell_json}')
        return

    # Full-grid mode (legacy; may hit the multi-fit hang)
    all_results = []
    for fs_name, (feature_cols, maxsize) in FEATURE_SETS.items():
        for op_name, op_config in OPERATOR_SETS.items():
            s = run_one_cell(df, TARGET, fs_name, feature_cols, maxsize,
                             op_name, op_config, groups=None)
            all_results.append(s)

    summary_df = build_summary_table(all_results)
    out_csv = Path(RESULTS_DIR) / f'pysr_grid_summary_{TARGET}.csv'
    summary_df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}  ({len(summary_df)} rows)')
    json_path = Path(RESULTS_DIR) / f'pysr_grid_full_{TARGET}.json'
    with open(json_path, 'w') as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
