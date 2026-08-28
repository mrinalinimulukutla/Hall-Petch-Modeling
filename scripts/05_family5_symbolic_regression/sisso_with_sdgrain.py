#!/usr/bin/env python3
"""
SISSO WITH SD_grain in the feature pool — fair-comparison variant (YS + HV).
============================================================================

Canonical SISSO (scripts/sisso_analysis.py) searches over composition
descriptors + d^(-1/2) ONLY, and only for YS. PySR's grid was given SD_GS
(grain-size standard deviation) as a first-class feature, for BOTH YS and HV.
To compare SISSO and PySR on the SAME footing, this script:

  1. Adds SD_GS to the SISSO SIS pool and re-runs the Full model for BOTH
     YS and HV (identical config: operators +-*/, tier 2, dimension 3,
     SIS top-k 20).
  2. Reports Train/LOO/LOBO R^2 and BIC (same compute_ic formula as canonical).
  3. External-validates each discovered equation on the SAME external dataset
     SISSO/PySR used (scripts/external_validation.py: 82 YS points, 25 HV
     points), with the SAME imputation as the PySR external validation:
       - SD_GS   <- linear fit SD_GS ~ GrainSize on training (r = 0.80)
       - processing (ColdWork/RecrystT/HoldTime) <- training medians
     and the SAME singularity audit (flag if |pred| > 3x training max).

It does NOT touch canonical results: writes only sisso_sdgrain_results.csv.
SISSO helpers are copied verbatim from sisso_analysis.py (importing that
module would trigger the canonical run and overwrite sisso_results.csv).

Output: results/sisso_sdgrain_results.csv  (one row per target)
"""
import re
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

from TorchSisso.FeatureSpaceConstruction import feature_space_construction
from TorchSisso.Regressor import Regressor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR
import external_validation as ev   # guarded by __main__, safe to import

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# ---- elemental property database (identical to sisso_analysis.py) ----
RADII = {'Al': 143, 'Co': 125, 'Cr': 128, 'Cu': 128, 'Fe': 126, 'Mn': 127, 'Ni': 124, 'V': 134}
VEC_VALS = {'Al': 3, 'Co': 9, 'Cr': 6, 'Cu': 11, 'Fe': 8, 'Mn': 7, 'Ni': 10, 'V': 5}
EN = {'Al': 1.61, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.90, 'Fe': 1.83, 'Mn': 1.55, 'Ni': 1.91, 'V': 1.63}
TM = {'Al': 933, 'Co': 1768, 'Cr': 2180, 'Cu': 1358, 'Fe': 1811, 'Mn': 1519, 'Ni': 1728, 'V': 2183}
SHEAR_MOD = {'Al': 26, 'Co': 75, 'Cr': 115, 'Cu': 48, 'Fe': 82, 'Mn': 79, 'Ni': 76, 'V': 47}
BULK_MOD = {'Al': 76, 'Co': 180, 'Cr': 160, 'Cu': 140, 'Fe': 170, 'Mn': 120, 'Ni': 180, 'V': 158}
A_FCC = {'Al': 4.050, 'Co': 3.545, 'Cr': 3.520, 'Cu': 3.615, 'Fe': 3.590, 'Mn': 3.540, 'Ni': 3.524, 'V': 3.720}
MASS = {'Al': 26.98, 'Co': 58.93, 'Cr': 52.00, 'Cu': 63.55, 'Fe': 55.85, 'Mn': 54.94, 'Ni': 58.69, 'V': 50.94}

OPERATORS = ['+', '-', '*', '/']
N_OPERATORS, DIMENSION, SIS_FEATURES = 2, 3, 20

FEAT_COMP_PHYSICS = [
    'r_mean', 'r_var', 'r_delta', 'r_range',
    'mu_mean', 'mu_var', 'mu_delta', 'mu_range',
    'EN_mean', 'EN_var', 'EN_delta', 'EN_range',
    'Tm_mean', 'Tm_var', 'Tm_delta', 'Tm_range',
    'VEC_mean', 'K_mean', 'K_var',
    'delta', 'dS_mix', 'dH_mix', 'Omega',
    'Phi_VLC', 'eps_Labusch', 'sigma_TC',
]
FEAT_POOL = FEAT_COMP_PHYSICS + ['d_inv_sqrt', 'SD_GS']   # <-- canonical + SD_GS

SAFE_FUNCS = {'square': lambda x: x ** 2, 'cube': lambda x: x ** 3,
              'sqrt': lambda x: np.sqrt(np.abs(x)), 'log': lambda x: np.log(np.abs(x) + 1e-12)}

print('=' * 72)
print('SISSO + SD_grain  (YS + HV; canonical results untouched)')
print('=' * 72)

# ---- load training data --------------------------------------------------
df = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
df = df.loc[:, ~df.columns.duplicated()].copy()
df['Omega'] = df['Omega'].clip(upper=100)
df = df.replace([np.inf, -np.inf], np.nan)

# SD_GS / processing imputation models (for external prediction) — same as PySR
_sd_fit = LinearRegression().fit(df[['GrainSize']].dropna().values,
                                 df.loc[df['GrainSize'].notna(), 'SD_GS'].values)
PROC_MEDIANS = {c: float(df[c].median()) for c in ['ColdWork', 'RecrystT', 'HoldTime']}


def compute_oliynyk_features(row):
    fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
    active = {el: c for el, c in fracs.items() if c > 0}
    features = {}
    properties = {'r': RADII, 'mu': SHEAR_MOD, 'K': BULK_MOD, 'EN': EN,
                  'Tm': TM, 'VEC': VEC_VALS, 'mass': MASS, 'a_fcc': A_FCC}
    for prop_name, prop_dict in properties.items():
        vals = np.array([prop_dict[el] for el in ELEMENTS])
        cs = np.array([fracs[el] for el in ELEMENTS])
        active_vals = [prop_dict[el] for el in active]
        mean_val = np.sum(cs * vals)
        features[f'{prop_name}_mean'] = mean_val
        var_val = np.sum(cs * (vals - mean_val) ** 2)
        features[f'{prop_name}_var'] = var_val
        features[f'{prop_name}_delta'] = np.sqrt(var_val) / abs(mean_val) if mean_val != 0 else 0.0
        features[f'{prop_name}_range'] = max(active_vals) - min(active_vals)
    return pd.Series(features)


# ---- SISSO helpers (verbatim from sisso_analysis.py) ---------------------
def run_sisso(df_input, label='SISSO'):
    print(f'\n  Running {label}...')
    fsc = feature_space_construction(operators=OPERATORS, df=df_input.copy(),
                                     no_of_operators=N_OPERATORS, device='cpu')
    x_expanded, y_tensor, names = fsc.feature_space()
    print(f'    Expanded feature space: {x_expanded.shape[1]} features')
    reg = Regressor(x_expanded, y_tensor, names, dimension=DIMENSION, sis_features=SIS_FEATURES)
    rmse, equation, r2, _ = reg.regressor_fit()
    print(f'    Best equation: {equation}')
    print(f'    Train R2: {r2:.4f}')
    return {'r2': r2, 'equation': equation, 'feature_names': names,
            'x_expanded': x_expanded, 'y_tensor': y_tensor}


def _eval_sisso_equation(equation_str, x_data, names):
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    name_to_idx = {name: i for i, name in enumerate(names)}
    eq = equation_str.strip()
    prediction = np.zeros(x_np.shape[0])
    term_pattern = r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*(.+?)(?=\s*[+-]\s*[\d.]|\s*$)'
    for coef_str, feat_name in re.findall(term_pattern, eq):
        coef = float(coef_str.replace(' ', ''))
        feat_name = feat_name.strip()
        if feat_name in name_to_idx:
            prediction += coef * x_np[:, name_to_idx[feat_name]]
        else:
            for name, idx in name_to_idx.items():
                if name.strip() == feat_name:
                    prediction += coef * x_np[:, idx]
                    break
    remaining = re.sub(term_pattern, '', eq).strip()
    if remaining:
        try:
            prediction += float(remaining.replace(' ', ''))
        except ValueError:
            pass
    return prediction[0] if prediction.shape[0] == 1 else prediction


def sisso_loo(x_expanded, y_tensor, names):
    n_samples = len(y_tensor)
    preds = np.zeros(n_samples)
    x_np = x_expanded.numpy() if isinstance(x_expanded, torch.Tensor) else x_expanded
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor
    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool); mask[i] = False
        reg = Regressor(torch.tensor(x_np[mask], dtype=torch.float32),
                        torch.tensor(y_np[mask], dtype=torch.float32),
                        names, dimension=DIMENSION, sis_features=SIS_FEATURES)
        _, equation, _, _ = reg.regressor_fit()
        preds[i] = _eval_sisso_equation(equation, x_np[i:i + 1], names)
    return r2_score(y_np, preds), np.sqrt(mean_squared_error(y_np, preds)), preds


def sisso_lobo(x_expanded, y_tensor, names, groups_arr):
    x_np = x_expanded.numpy() if isinstance(x_expanded, torch.Tensor) else x_expanded
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor
    preds = np.zeros(len(y_np))
    for tr, te in LeaveOneGroupOut().split(x_np, y_np, groups_arr):
        reg = Regressor(torch.tensor(x_np[tr], dtype=torch.float32),
                        torch.tensor(y_np[tr], dtype=torch.float32),
                        names, dimension=DIMENSION, sis_features=SIS_FEATURES)
        _, equation, _, _ = reg.regressor_fit()
        preds[te] = _eval_sisso_equation(equation, x_np[te], names)
    return r2_score(y_np, preds)


def compute_bic(y_true, y_pred, k):
    n = len(y_true)
    rss = max(np.sum((y_true - y_pred) ** 2), 1e-15)
    return n * np.log(rss / n) + k * np.log(n)


# ---- external feature frame (computed identically for YS and HV) ---------
def build_external_features(ext_df):
    """Every base feature a SISSO equation might reference, per external row."""
    rows = []
    for _, r in ext_df.iterrows():
        oli = compute_oliynyk_features(r)
        fracs = {el: float(r.get(f'{el}_frac', 0.0) or 0.0) for el in ELEMENTS}
        desc = ev.compute_hea_descriptors(fracs)
        gs = float(r['GrainSize'])
        feat = oli.to_dict()
        feat.update({
            'delta': desc['delta'], 'dS_mix': desc['dS_mix'], 'dH_mix': desc['dH_mix'],
            'Omega': min(desc['Omega'], 100.0), 'Phi_VLC': desc['Phi_VLC'],
            'eps_Labusch': desc['eps_Labusch'], 'sigma_TC': desc['sigma_TC'],
            'd_inv_sqrt': gs ** -0.5,
            'SD_GS': float(_sd_fit.predict([[gs]])[0]),
        })
        for c in ['ColdWork', 'RecrystT', 'HoldTime']:
            feat[c] = PROC_MEDIANS[c]
        rows.append(feat)
    return pd.DataFrame(rows, index=ext_df.index)


def external_score(eq_str, feat_df, y_ext, extreme_thresh):
    ns = {c: feat_df[c].values.astype(float) for c in feat_df.columns}
    ns.update(SAFE_FUNCS)
    try:
        preds = np.asarray(eval(eq_str, {'__builtins__': {}}, ns), dtype=float)
        if preds.ndim == 0:
            preds = np.full(len(feat_df), float(preds))
    except Exception as e:
        return dict(Ext_R2=np.nan, Ext_RMSE=np.nan, safe=False, note=f'eval failed: {e}')
    finite = np.isfinite(preds)
    n_nonfinite = int((~finite).sum())
    extreme = finite & (np.abs(preds) > extreme_thresh)
    safe = (n_nonfinite == 0) and (int(extreme.sum()) == 0)
    r2 = rmse = np.nan
    if finite.sum() >= 3:
        r2 = r2_score(y_ext[finite], preds[finite])
        rmse = float(np.sqrt(mean_squared_error(y_ext[finite], preds[finite])))
    note = '' if safe else f'{n_nonfinite} non-finite, {int(extreme.sum())} extreme (singularity)'
    return dict(Ext_R2=r2, Ext_RMSE=rmse, safe=safe, note=note)


# ---- external dataset (same 82-pt set; HV truth on 25 Huang points) ------
ext = ev.load_all_external_data(df.dropna(subset=['YS']).reset_index(drop=True))
ext_feat = build_external_features(ext)


# ============================================================
# RUN both targets identically
# ============================================================
def run_target(target):
    sub = df.dropna(subset=[target]).copy()
    y = sub[target].values.astype(float)
    n = len(y)
    groups = sub['Iteration'].values
    sub['d_inv_sqrt'] = sub['GrainSize'].values ** -0.5
    oli = sub.apply(compute_oliynyk_features, axis=1)
    for c in oli.columns:
        sub[c] = oli[c].values

    print(f'\n{"="*72}\nTARGET = {target}  (n={n})\n{"="*72}')
    df_in = pd.DataFrame({target: y})
    for feat in FEAT_POOL:
        vals = sub[feat].fillna(0).values
        if np.std(vals) > 1e-12:
            df_in[feat] = vals

    res = run_sisso(df_in, label=f'Full + SD_grain ({target})')
    print('  LOO...'); t = time.time()
    r2_loo, rmse_loo, _ = sisso_loo(res['x_expanded'], res['y_tensor'], res['feature_names'])
    print(f'    LOO R2={r2_loo:.4f} ({time.time()-t:.0f}s)')
    print('  LOBO...'); t = time.time()
    r2_lobo = sisso_lobo(res['x_expanded'], res['y_tensor'], res['feature_names'], groups)
    print(f'    LOBO R2={r2_lobo:.4f} ({time.time()-t:.0f}s)')

    k = DIMENSION + 1
    y_pred_train = _eval_sisso_equation(res['equation'], res['x_expanded'], res['feature_names'])
    bic = compute_bic(y, y_pred_train, k)

    # external — same set/imputation/audit as PySR
    truth_col = 'YS_exp' if target == 'YS' else 'HV_exp'
    mask = ext[truth_col].notna().values
    y_ext = ext.loc[mask, truth_col].values.astype(float)
    es = external_score(res['equation'], ext_feat.loc[mask].reset_index(drop=True),
                        y_ext, extreme_thresh=3.0 * float(np.max(np.abs(y))))

    uses_sd = 'SD_GS' in res['equation']
    print(f'  uses SD_GS: {uses_sd} | BIC={bic:.1f} | Ext_RMSE={es["Ext_RMSE"]:.1f} '
          f'(n={len(y_ext)}) safe={es["safe"]}')
    return {
        'Model': f'SISSO Full + SD_grain ({target})', 'Target': target,
        'Train_R2': round(float(res['r2']), 4), 'LOO_R2': round(r2_loo, 4),
        'LOO_RMSE': round(rmse_loo, 2), 'LOBO_R2': round(r2_lobo, 4),
        'BIC': round(bic, 2), 'k_eff': k, 'uses_SD_GS': uses_sd,
        'Ext_R2': round(es['Ext_R2'], 4) if np.isfinite(es['Ext_R2']) else '',
        'Ext_RMSE': round(es['Ext_RMSE'], 1) if np.isfinite(es['Ext_RMSE']) else '',
        'n_ext': len(y_ext),
        'Singularity_safe': 'yes' if es['safe'] else 'no',
        'imputed_features': 'SD_GS+ColdWork+RecrystT+HoldTime',
        'note': es['note'],
        'equation': res['equation'],
    }


rows = [run_target('YS'), run_target('HV')]
out = pd.DataFrame(rows)
out.to_csv(f'{RESULTS_DIR}/sisso_sdgrain_results.csv', index=False)
print(f'\nWrote results/sisso_sdgrain_results.csv')

print('\n' + '=' * 72)
print('SUMMARY — SISSO + SD_grain vs canonical & PySR (same external set)')
print('=' * 72)
for r in rows:
    print(f"  {r['Model']:30s}  LOO={r['LOO_R2']:.3f}  LOBO={r['LOBO_R2']:.3f}  "
          f"BIC={r['BIC']:.1f}  Ext_RMSE={r['Ext_RMSE']}  safe={r['Singularity_safe']}  SD_GS={r['uses_SD_GS']}")
print('  ' + '-' * 68)
print('  Canonical SISSO Full (YS, no SD): LOO=0.665  BIC=714.0  Ext_RMSE=420.6  safe=no')
print('  SISSO Robust          (YS, no SD): LOO=0.609  BIC=716.6  Ext_RMSE=162.9  safe=yes')
print('=' * 72)
