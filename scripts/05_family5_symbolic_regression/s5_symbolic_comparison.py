#!/usr/bin/env python3
"""
S5 symbolic-regression tier: PySR vs SISSO on the SAME curated-Wen inputs.
==========================================================================

S5 replaces hand-engineered interaction features with symbolic regression that
DISCOVERS the form. To compare the two engines on equal terms, both search the
identical curated-Wen feature pool (the F4 set):

  F4 = curated Wen [VEC, dH_mix, dS_mix, Omega, delta_chi, delta]
       + processing [ColdWork, RecrystT, HoldTime]
       + grain [d_inv_sqrt, SD_GS]                       (11 features)

  - PySR side : already computed by scripts/pysr_grid_analysis.py on this exact
    set (feature_set 'F3_wen'); read from results/pysr_grid_summary_{YS,HV}.csv.
  - SISSO side: this script runs SISSO on the SAME 11-feature pool (operators
    +-*/, tier 2, dimension 3 -> k_eff 4 ~ PySR O2 level), for YS and HV.

Both report LOO, LOBO, BIC, and external-set RMSE on the same 82-point literature
set (25 HV points), with SD_GS / processing imputed identically (the PySR external
numbers come from results/pysr_external_validation.csv). The common currency for
matching the two engines is k_eff + BIC + LOO/LOBO + external RMSE — NOT node
count vs dimension, which are not comparable units.

Canonical SISSO results are untouched; output is results/s5_symbolic_comparison.csv.
"""
import re
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

from TorchSisso.FeatureSpaceConstruction import feature_space_construction
from TorchSisso.Regressor import Regressor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR
import external_validation as ev   # guarded by __main__, safe to import

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# F4 = curated Wen + processing + grain (== pysr_grid 'F3_wen' feature set)
FEAT_POOL = ['VEC', 'dH_mix', 'dS_mix', 'Omega', 'delta_chi', 'delta',
             'ColdWork', 'RecrystT', 'HoldTime', 'd_inv_sqrt', 'SD_GS']
OPERATORS = ['+', '-', '*', '/']
N_OPERATORS, DIMENSION, SIS_FEATURES = 2, 3, 20

SAFE_FUNCS = {'square': lambda x: x ** 2, 'cube': lambda x: x ** 3,
              'sqrt': lambda x: np.sqrt(np.abs(x)), 'log': lambda x: np.log(np.abs(x) + 1e-12)}

print('=' * 72)
print('S5 SYMBOLIC COMPARISON — PySR vs SISSO on curated-Wen (F4) inputs')
print('=' * 72)

df = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
df = df.loc[:, ~df.columns.duplicated()].copy()
df['Omega'] = df['Omega'].clip(upper=100)
df = df.replace([np.inf, -np.inf], np.nan)
df['d_inv_sqrt'] = df['GrainSize'].values ** -0.5

_sd_fit = LinearRegression().fit(df[['GrainSize']].dropna().values,
                                 df.loc[df['GrainSize'].notna(), 'SD_GS'].values)
PROC_MEDIANS = {c: float(df[c].median()) for c in ['ColdWork', 'RecrystT', 'HoldTime']}


# ---- SISSO helpers (verbatim pattern from sisso_analysis.py) -------------
def run_sisso(df_input):
    fsc = feature_space_construction(operators=OPERATORS, df=df_input.copy(),
                                     no_of_operators=N_OPERATORS, device='cpu')
    x_expanded, y_tensor, names = fsc.feature_space()
    reg = Regressor(x_expanded, y_tensor, names, dimension=DIMENSION, sis_features=SIS_FEATURES)
    rmse, equation, r2, _ = reg.regressor_fit()
    return {'r2': r2, 'equation': equation, 'feature_names': names,
            'x_expanded': x_expanded, 'y_tensor': y_tensor}


def _eval(equation_str, x_data, names):
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    name_to_idx = {n: i for i, n in enumerate(names)}
    pred = np.zeros(x_np.shape[0])
    pat = r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*(.+?)(?=\s*[+-]\s*[\d.]|\s*$)'
    for coef_str, fn in re.findall(pat, equation_str.strip()):
        coef = float(coef_str.replace(' ', '')); fn = fn.strip()
        idx = name_to_idx.get(fn)
        if idx is None:
            for nm, i in name_to_idx.items():
                if nm.strip() == fn:
                    idx = i; break
        if idx is not None:
            pred += coef * x_np[:, idx]
    rem = re.sub(pat, '', equation_str.strip()).strip()
    if rem:
        try:
            pred += float(rem.replace(' ', ''))
        except ValueError:
            pass
    return pred[0] if pred.shape[0] == 1 else pred


def sisso_loo(x_expanded, y_tensor, names):
    x_np = x_expanded.numpy(); y_np = y_tensor.numpy()
    preds = np.zeros(len(y_np))
    for i in range(len(y_np)):
        m = np.ones(len(y_np), bool); m[i] = False
        reg = Regressor(torch.tensor(x_np[m], dtype=torch.float32),
                        torch.tensor(y_np[m], dtype=torch.float32),
                        names, dimension=DIMENSION, sis_features=SIS_FEATURES)
        _, eq, _, _ = reg.regressor_fit()
        preds[i] = _eval(eq, x_np[i:i + 1], names)
    return r2_score(y_np, preds), np.sqrt(mean_squared_error(y_np, preds))


def sisso_lobo(x_expanded, y_tensor, names, groups):
    x_np = x_expanded.numpy(); y_np = y_tensor.numpy()
    preds = np.zeros(len(y_np))
    for tr, te in LeaveOneGroupOut().split(x_np, y_np, groups):
        reg = Regressor(torch.tensor(x_np[tr], dtype=torch.float32),
                        torch.tensor(y_np[tr], dtype=torch.float32),
                        names, dimension=DIMENSION, sis_features=SIS_FEATURES)
        _, eq, _, _ = reg.regressor_fit()
        preds[te] = _eval(eq, x_np[te], names)
    return r2_score(y_np, preds)


def bic_from_pred(y, yp, k):
    n = len(y); rss = max(np.sum((y - yp) ** 2), 1e-15)
    return n * np.log(rss / n) + k * np.log(n)


# ---- external feature frame (curated-Wen pool, imputed like PySR) --------
ext = ev.load_all_external_data(df.dropna(subset=['YS']).reset_index(drop=True))


def build_ext_features(ext_df):
    rows = []
    for _, r in ext_df.iterrows():
        fracs = {el: float(r.get(f'{el}_frac', 0.0) or 0.0) for el in ELEMENTS}
        desc = ev.compute_hea_descriptors(fracs)
        gs = float(r['GrainSize'])
        feat = {'VEC': desc['VEC'], 'dH_mix': desc['dH_mix'], 'dS_mix': desc['dS_mix'],
                'Omega': min(desc['Omega'], 100.0), 'delta_chi': desc['delta_chi'], 'delta': desc['delta'],
                'd_inv_sqrt': gs ** -0.5, 'SD_GS': float(_sd_fit.predict([[gs]])[0])}
        for c in ['ColdWork', 'RecrystT', 'HoldTime']:
            feat[c] = PROC_MEDIANS[c]
        rows.append(feat)
    return pd.DataFrame(rows, index=ext_df.index)


ext_feat = build_ext_features(ext)


def ext_score(eq, feat_df, y_ext, thr):
    ns = {c: feat_df[c].values.astype(float) for c in feat_df.columns}; ns.update(SAFE_FUNCS)
    try:
        p = np.asarray(eval(eq, {'__builtins__': {}}, ns), dtype=float)
        if p.ndim == 0:
            p = np.full(len(feat_df), float(p))
    except Exception as e:
        return np.nan, np.nan, False, f'eval failed: {e}'
    fin = np.isfinite(p); extr = fin & (np.abs(p) > thr)
    safe = (int((~fin).sum()) == 0) and (int(extr.sum()) == 0)
    if fin.sum() >= 3:
        return r2_score(y_ext[fin], p[fin]), float(np.sqrt(mean_squared_error(y_ext[fin], p[fin]))), safe, ''
    return np.nan, np.nan, safe, 'too few finite'


# ---- run SISSO on F4 for each target -------------------------------------
def run_sisso_target(target):
    sub = df.dropna(subset=[target]).copy()
    y = sub[target].values.astype(float)
    groups = sub['Iteration'].values
    print(f'\n--- SISSO-F4 ({target}, n={len(y)}) ---')
    din = pd.DataFrame({target: y})
    for f in FEAT_POOL:
        v = sub[f].fillna(0).values
        if np.std(v) > 1e-12:
            din[f] = v
    res = run_sisso(din)
    r2_loo, _ = sisso_loo(res['x_expanded'], res['y_tensor'], res['feature_names'])
    r2_lobo = sisso_lobo(res['x_expanded'], res['y_tensor'], res['feature_names'], groups)
    k = DIMENSION + 1
    bic = bic_from_pred(y, _eval(res['equation'], res['x_expanded'], res['feature_names']), k)
    tcol = 'YS_exp' if target == 'YS' else 'HV_exp'
    mask = ext[tcol].notna().values
    er2, ermse, safe, note = ext_score(res['equation'], ext_feat.loc[mask].reset_index(drop=True),
                                       ext[tcol].dropna().values.astype(float), 3.0 * float(np.max(np.abs(y))))
    print(f'  eq: {res["equation"]}')
    print(f'  LOO={r2_loo:.3f} LOBO={r2_lobo:.3f} BIC={bic:.1f} ExtRMSE={ermse:.1f} safe={safe} usesSD={"SD_GS" in res["equation"]}')
    return {'Engine': 'SISSO', 'Target': target, 'FeatureSet': 'F4 curated-Wen', 'k_eff': k,
            'LOO_R2': round(r2_loo, 4), 'LOBO_R2': round(r2_lobo, 4), 'BIC': round(bic, 1),
            'Ext_R2': round(er2, 4) if np.isfinite(er2) else '', 'Ext_RMSE': round(ermse, 1) if np.isfinite(ermse) else '',
            'Singularity_safe': 'yes' if safe else 'no', 'uses_SD_GS': 'SD_GS' in res['equation'],
            'equation': res['equation']}


# ---- pull existing PySR-F4 (F3_wen) rows for the matched comparison ------
def pysr_rows(target):
    g = pd.read_csv(f'{RESULTS_DIR}/pysr_grid_summary_{target}.csv')
    g = g[g.feature_set == 'F3_wen']
    pe = pd.read_csv(f'{RESULTS_DIR}/pysr_external_validation.csv')
    out = []
    for _, r in g.iterrows():
        ev_row = pe[(pe.Target == target) & (pe.feature_set == 'F3_wen') &
                    (pe.op_set == r.op_set) & (pe.selection == r.selection)]
        ext_rmse = ext_safe = ''
        if len(ev_row):
            ext_rmse = ev_row.iloc[0]['Ext_RMSE_MPa']; ext_safe = ev_row.iloc[0]['Singularity_safe']
        bic = pe[(pe.Target == target) & (pe.feature_set == 'F3_wen') &
                 (pe.op_set == r.op_set) & (pe.selection == r.selection)]
        bic_val = bic.iloc[0]['BIC'] if len(bic) else ''
        out.append({'Engine': f'PySR ({r.op_set},{r.selection})', 'Target': target,
                    'FeatureSet': 'F4 curated-Wen', 'k_eff': int(r.n_constants),
                    'LOO_R2': round(float(r.cv_refit_LOO_R2), 4), 'LOBO_R2': round(float(r.cv_refit_LOBO_R2), 4),
                    'BIC': bic_val, 'Ext_R2': '', 'Ext_RMSE': ext_rmse, 'Singularity_safe': ext_safe,
                    'uses_SD_GS': 'SD_GS' in str(r.equation), 'equation': r.equation})
    return out


rows = []
for tgt in ['YS', 'HV']:
    rows.append(run_sisso_target(tgt))
    rows.extend(pysr_rows(tgt))

out = pd.DataFrame(rows)
out.to_csv(f'{RESULTS_DIR}/s5_symbolic_comparison.csv', index=False)
print(f'\nWrote results/s5_symbolic_comparison.csv ({len(out)} rows)')

print('\n' + '=' * 72)
print('S5 — PySR vs SISSO on identical curated-Wen (F4) inputs')
print('=' * 72)
for tgt in ['YS', 'HV']:
    print(f'\n{tgt}:')
    sub = out[out.Target == tgt].sort_values('LOO_R2', ascending=False)
    for _, r in sub.iterrows():
        print(f"  {r.Engine:22s} k={r.k_eff} LOO={r.LOO_R2:+.3f} LOBO={r.LOBO_R2:+.3f} "
              f"BIC={r.BIC} ExtRMSE={r.Ext_RMSE} safe={r.Singularity_safe} SD={r.uses_SD_GS}")
