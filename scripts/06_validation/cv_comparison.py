#!/usr/bin/env python3
"""
3-Way Cross-Validation Comparison: 5-fold vs LOO vs LOBO
=========================================================
Reports all three protocols side-by-side for every headline model that
is cheap to refit. Targets the gap the integration was designed to expose:

  conference deck → 5-fold (within-batch, k=5)
  Integration → LOO  (within-batch, k=n)
  Integration → LOBO (cross-batch, k=6 = number of campaigns)

5-fold and LOO are both within-batch random shuffles. The headline
generalization story is the LOBO drop, not the LOO drop. This script lets
us report both honestly.

Models covered:
  - Classical Hall-Petch (1/sqrt(d) only)
  - Power-law Hall-Petch (free exponent)
  - M3 composition-dependent HP (7 elements + d^-1/2)
  - PCA-OLS (curated Wen + SD_grain → 6 PCs)
  - SISSO Full (paper Eq. 4 features, refit with linear regression on the
    same composite features so we get a fair 5-fold for it)
  - SISSO Robust (paper Eq. 5 features, same refit approach)
  - Compact HV elbow (form-frozen curve_fit refit on the published features)
  - compact-equation stream YS compact (form-frozen curve_fit refit on the published features)

Output: results/cv_comparison.csv with columns:
  Model, Family, n_params, R2_5fold, R2_LOO, R2_LOBO,
  RMSE_5fold, RMSE_LOO, RMSE_LOBO, Notes
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
RNG_SEED = 42

# ============================================================
# 1. LOAD DATA
# ============================================================
print('=' * 70)
print('3-WAY CV COMPARISON: 5-fold vs LOO vs LOBO')
print('=' * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv').dropna(subset=['YS']).reset_index(drop=True)
df_hv = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv').dropna(subset=['HV']).reset_index(drop=True)
groups = df['Iteration'].values
groups_hv = df_hv['Iteration'].values
n = len(df)
print(f'n = {n} alloys with YS; {len(df_hv)} alloys with HV')
print(f'batches: {sorted(set(groups))}')


# ============================================================
# 2. HELPER: REFIT-CV WITH ANY ESTIMATOR + FORM
# ============================================================
def refit_linear_cv(X, y, groups=None):
    """Returns (r2_5fold, rmse_5fold, r2_loo, rmse_loo, r2_lobo, rmse_lobo)."""
    out = {}
    splitters = {
        '5fold': (KFold(n_splits=5, shuffle=True, random_state=RNG_SEED), False),
        'LOO':   (LeaveOneOut(), False),
        'LOBO':  (LeaveOneGroupOut(), True),
    }
    for tag, (splitter, needs_groups) in splitters.items():
        preds = np.full_like(y, np.nan, dtype=float)
        if needs_groups and groups is None:
            out[tag] = (np.nan, np.nan)
            continue
        iter_args = (X, y, groups) if needs_groups else (X,)
        for tr, te in splitter.split(*iter_args):
            m = LinearRegression().fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        ok = np.isfinite(preds)
        out[tag] = (
            float(r2_score(y[ok], preds[ok])),
            float(np.sqrt(mean_squared_error(y[ok], preds[ok]))),
        )
    return out


def refit_curve_cv(form, X_pack, y, p0, groups=None):
    """For symbolic forms with named constants. X_pack is a tuple of arrays."""
    out = {}
    splitters = {
        '5fold': (KFold(n_splits=5, shuffle=True, random_state=RNG_SEED), False),
        'LOO':   (LeaveOneOut(), False),
        'LOBO':  (LeaveOneGroupOut(), True),
    }
    n_local = len(y)
    for tag, (splitter, needs_groups) in splitters.items():
        preds = np.full(n_local, np.nan)
        if needs_groups and groups is None:
            out[tag] = (np.nan, np.nan)
            continue
        iter_args = (X_pack[0], y, groups) if needs_groups else (X_pack[0],)
        for tr, te in splitter.split(*iter_args):
            try:
                popt, _ = curve_fit(form,
                                    tuple(arr[tr] for arr in X_pack),
                                    y[tr], p0=p0, maxfev=20_000)
                preds[te] = form(tuple(arr[te] for arr in X_pack), *popt)
            except Exception:
                preds[te] = np.nan
        ok = np.isfinite(preds)
        if ok.sum() < 5:
            out[tag] = (np.nan, np.nan)
            continue
        out[tag] = (
            float(r2_score(y[ok], preds[ok])),
            float(np.sqrt(mean_squared_error(y[ok], preds[ok]))),
        )
    return out


# ============================================================
# 3. PER-MODEL FITS
# ============================================================
rows = []

# Common YS targets
y = df['YS'].values.astype(float)
d_inv_sqrt = df['d_inv_sqrt'].values.astype(float)

# --- Classical HP (1 feature) ---
X = d_inv_sqrt.reshape(-1, 1)
r = refit_linear_cv(X, y, groups)
rows.append({
    'Model': 'Classical HP', 'Family': 'classical', 'Target': 'YS',
    'n_params': 2,
    'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
    'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
    'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
    'Notes': 'YS = sigma_0 + k * d^-1/2 (Hall-Petch baseline)',
})
print(f"\nClassical HP   YS  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- Power-law HP (free exponent) ---
def hp_power(X, sigma0, k, n_exp):
    d = X[0]
    return sigma0 + k * d ** (-n_exp)

X_pack = (df['GrainSize'].values.astype(float),)
r = refit_curve_cv(hp_power, X_pack, y, p0=[100, 700, 0.5], groups=groups)
rows.append({
    'Model': 'Power-law HP', 'Family': 'classical', 'Target': 'YS',
    'n_params': 3,
    'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
    'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
    'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
    'Notes': 'YS = sigma_0 + k * d^-n with n fitted (3 params)',
})
print(f"Power-law HP   YS  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- M3 composition-dependent HP ---
elem_cols_no_ni = [f'{e}_frac' for e in ELEMENTS if e != 'Ni']
X = np.column_stack([df[elem_cols_no_ni].values.astype(float),
                     d_inv_sqrt.reshape(-1, 1)])
r = refit_linear_cv(X, y, groups)
rows.append({
    'Model': 'M3 (sigma_0(7 elem) + k*d^-1/2)', 'Family': 'linear', 'Target': 'YS',
    'n_params': 9,
    'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
    'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
    'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
    'Notes': 'sigma_0 linear in 7 element fractions; constant k_HP',
})
print(f"M3             YS  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- PCA-OLS on curated Wen + SD_grain ---
curated = ['VEC', 'dH_mix', 'dS_mix', 'Omega', 'delta_chi', 'delta',
           'd_inv_sqrt', 'SD_GS', 'ColdWork', 'RecrystT', 'HoldTime']
curated = [c for c in curated if c in df.columns]
X_raw = df[curated].values.astype(float)
X_std = StandardScaler().fit_transform(X_raw)
Z = PCA(n_components=6).fit_transform(X_std)
r = refit_linear_cv(Z, y, groups)
rows.append({
    'Model': 'PCA-OLS (curated Wen + SD_grain, 6 PCs)', 'Family': 'linear',
    'Target': 'YS', 'n_params': 7,
    'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
    'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
    'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
    'Notes': 'PCA-OLS baseline; 6 PCs retain ~88% input variance',
})
print(f"PCA-OLS        YS  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- SISSO Full: linear regression on the 3 SISSO composite features ---
# sigma_y = c1 * (r_var/r_range) + c2 * (d_inv_sqrt/dS_mix) + c3 * (EN_var/delta_mu) + c0
# Build these features ourselves so we can refit per fold.
if {'r_var', 'r_range', 'dS_mix', 'EN_var', 'delta_mu'} <= set(df.columns):
    f1 = df['r_var'].values.astype(float) / df['r_range'].values.astype(float)
    f2 = df['d_inv_sqrt'].values.astype(float) / df['dS_mix'].values.astype(float)
    f3 = df['EN_var'].values.astype(float) / df['delta_mu'].values.astype(float)
    X = np.column_stack([f1, f2, f3])
    r = refit_linear_cv(X, y, groups)
    rows.append({
        'Model': 'SISSO Full (refit)', 'Family': 'symbolic', 'Target': 'YS',
        'n_params': 4,
        'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
        'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
        'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
        'Notes': 'Refit constants on SISSO Full (Eq. 4) composite features',
    })
    print(f"SISSO Full     YS  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- SISSO Robust: linear regression on its 3 composite features ---
if {'sigma2_r', 'r_range', 'd_inv_sqrt', 'dS_mix', 'sigma2_chi', 'Phi_VLC'} <= set(df.columns):
    f1 = df['sigma2_r'].values.astype(float) / df['r_range'].values.astype(float)
    f2 = df['d_inv_sqrt'].values.astype(float) / df['dS_mix'].values.astype(float)
    f3 = df['sigma2_chi'].values.astype(float) - df['Phi_VLC'].values.astype(float)
    X = np.column_stack([f1, f2, f3])
    r = refit_linear_cv(X, y, groups)
    rows.append({
        'Model': 'SISSO Robust (refit)', 'Family': 'symbolic', 'Target': 'YS',
        'n_params': 4,
        'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
        'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
        'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
        'Notes': 'Refit constants on SISSO Robust (Eq. 5) composite features (bounded)',
    })
    print(f"SISSO Robust   YS  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- compact-equation stream YS compact equation ---
if {'VEC', 'dH_mix', 'SD_GS', 'GrainSize', 'delta_chi'} <= set(df.columns):
    def ys_compact(X, c1, c2, c3):
        VEC, dH, SD, d, dchi = X
        return VEC * (c1 * dH * SD / (d ** 2 * dchi) + c2 / dchi + c3)
    X_pack = (df['VEC'].values.astype(float),
              df['dH_mix'].values.astype(float),
              df['SD_GS'].values.astype(float),
              df['GrainSize'].values.astype(float),
              df['delta_chi'].values.astype(float))
    r = refit_curve_cv(ys_compact, X_pack, y, p0=[4.29, -2.13, 56.06], groups=groups)
    rows.append({
        'Model': 'Compact YS equation (refit)', 'Family': 'symbolic',
        'Target': 'YS', 'n_params': 4,
        'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
        'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
        'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
        'Notes': 'Form-frozen refit of YS = VEC*(c1*dH*SD/(d^2*dchi) + c2/dchi + c3)',
    })
    print(f"Compact YS eq  YS  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- Compact HV elbow equation ---
y_hv = df_hv['HV'].values.astype(float)
if {'GrainSize', 'SD_GS', 'dH_mix', 'HoldTime'} <= set(df_hv.columns):
    def hv_elbow(X, c0, c1, c2):
        d, SD, dH, t = X
        return c0 + c1 * (6.93 - d) / SD + c2 * dH / (t ** 2)
    X_pack = (df_hv['GrainSize'].values.astype(float),
              df_hv['SD_GS'].values.astype(float),
              df_hv['dH_mix'].values.astype(float),
              df_hv['HoldTime'].values.astype(float))
    r = refit_curve_cv(hv_elbow, X_pack, y_hv, p0=[221.46, -83.95, 1.0],
                       groups=groups_hv)
    rows.append({
        'Model': 'Compact HV elbow (refit)', 'Family': 'symbolic',
        'Target': 'HV', 'n_params': 3,
        'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
        'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
        'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
        'Notes': 'Form-frozen refit of HV = c0 + c1*(6.93-d)/SD + c2*dH/t^2',
    })
    print(f"Compact HV elb  HV  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# --- HV baseline Hall-Petch ---
X = df_hv['d_inv_sqrt'].values.astype(float).reshape(-1, 1)
r = refit_linear_cv(X, y_hv, groups_hv)
rows.append({
    'Model': 'Classical HP', 'Family': 'classical', 'Target': 'HV',
    'n_params': 2,
    'R2_5fold': r['5fold'][0], 'RMSE_5fold': r['5fold'][1],
    'R2_LOO':   r['LOO'][0],   'RMSE_LOO':   r['LOO'][1],
    'R2_LOBO':  r['LOBO'][0],  'RMSE_LOBO':  r['LOBO'][1],
    'Notes': 'HV = H_0 + k_H * d^-1/2',
})
print(f"HV baseline    HV  5f={r['5fold'][0]:.3f}  LOO={r['LOO'][0]:.3f}  LOBO={r['LOBO'][0]:.3f}")

# ============================================================
# 4. SAVE
# ============================================================
out = pd.DataFrame(rows)
out_path = f'{RESULTS_DIR}/cv_comparison.csv'
out.to_csv(out_path, index=False)
print(f'\nWrote {out_path}  ({len(out)} rows)')
print('\nFull table:')
print(out[['Model', 'Target', 'n_params',
           'R2_5fold', 'R2_LOO', 'R2_LOBO']].to_string(index=False))
