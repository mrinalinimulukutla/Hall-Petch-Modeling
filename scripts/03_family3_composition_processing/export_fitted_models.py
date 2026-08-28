"""Export fitted models and their coefficients for downstream verification.

Some claims in the paper (e.g., M3's alpha_V = +291 MPa) depend on
coefficients that live inside sklearn objects which the main analysis
scripts don't dump to disk. This exporter refits the small, fast linear
models from data and writes:

  results/m3_coefficients.csv         Long-format CSV with one row per
                                       coefficient (model, name, value,
                                       std_err, lo95, hi95).
  results/m3_model.pkl                 Joblib pickle of the fitted M3
                                       sklearn LinearRegression object
                                       (loadable for prediction/inspection).
  results/hv_baseline_coefficients.csv H_0, k_H, n_opt for HV Hall-Petch.
  results/hv_baseline_model.pkl        Joblib pickle of the HV HP fit.

Reviewers can load the pickles directly:

    from joblib import load
    m3 = load('results/m3_model.pkl')
    print(m3.coef_)         # 7 alpha_i values + k_HP
    print(m3.intercept_)    # sigma_00

Run time: ~5 s. Does not require PyMC, Optuna, or any heavy ML deps.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RESULTS_DIR

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import minimize_scalar

# Ni is the solvent and is absorbed into the intercept; the other 7 elements
# enter the composition descriptor vector for M3.
NON_NI_ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'V']

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(DATA_DIR / 'data_with_descriptors.csv')
df_ys = df.dropna(subset=['YS']).reset_index(drop=True)
df_hv = df.dropna(subset=['HV']).reset_index(drop=True)

print(f"Loaded {len(df_ys)} alloys with YS, {len(df_hv)} with HV")

# ============================================================
# Fit M3: sigma_y = sigma_00 + sum_i alpha_i * x_i + k_HP * d^(-1/2)
# ============================================================
X_m3 = np.column_stack(
    [df_ys[f'{el}_frac'].values for el in NON_NI_ELEMENTS]
    + [df_ys['d_inv_sqrt'].values]
)
y_ys = df_ys['YS'].values

m3 = LinearRegression().fit(X_m3, y_ys)
y_pred_m3 = m3.predict(X_m3)
r2_m3 = m3.score(X_m3, y_ys)

# LOO via hat matrix (analytical, exact for OLS)
X_aug = np.column_stack([np.ones(len(y_ys)), X_m3])  # include intercept
H = X_aug @ np.linalg.solve(X_aug.T @ X_aug, X_aug.T)
h_ii = np.diag(H)
resid = y_ys - y_pred_m3
loo_resid = resid / (1 - h_ii)
r2_loo_m3 = 1 - np.sum(loo_resid**2) / np.sum((y_ys - y_ys.mean())**2)

# OLS standard errors (homoscedastic) for each coefficient
sigma2 = np.sum(resid**2) / (len(y_ys) - X_aug.shape[1])
cov = sigma2 * np.linalg.inv(X_aug.T @ X_aug)
se = np.sqrt(np.diag(cov))

# t-quantile (n - p degrees of freedom) at 95% CI
from scipy.stats import t as t_dist
dof = len(y_ys) - X_aug.shape[1]
t_crit = t_dist.ppf(0.975, dof)

# M3 coefficient names in the same order as X_aug columns
coef_names = ['intercept (sigma_00)'] + [f'alpha_{el}' for el in NON_NI_ELEMENTS] + ['k_HP']
# Full coefficient vector with intercept first
beta_full = np.concatenate([[m3.intercept_], m3.coef_])

rows = []
for name, b, s in zip(coef_names, beta_full, se):
    rows.append({
        'model': 'M3',
        'coefficient': name,
        'value': b,
        'std_err': s,
        'lo95': b - t_crit * s,
        'hi95': b + t_crit * s,
        'units': 'MPa',
    })

# Headline metrics
rows.append({'model': 'M3', 'coefficient': 'R2_train',  'value': r2_m3,     'std_err': np.nan, 'lo95': np.nan, 'hi95': np.nan, 'units': ''})
rows.append({'model': 'M3', 'coefficient': 'R2_LOO',    'value': r2_loo_m3, 'std_err': np.nan, 'lo95': np.nan, 'hi95': np.nan, 'units': ''})
rows.append({'model': 'M3', 'coefficient': 'n_samples', 'value': len(y_ys), 'std_err': np.nan, 'lo95': np.nan, 'hi95': np.nan, 'units': ''})

m3_df = pd.DataFrame(rows)
m3_df.to_csv(RESULTS_DIR / 'm3_coefficients.csv', index=False)
dump(m3, RESULTS_DIR / 'm3_model.pkl')

print(f"\nM3 fitted:")
print(f"  intercept (sigma_00) = {m3.intercept_:+.1f}")
for el, alpha, s in zip(NON_NI_ELEMENTS, m3.coef_[:7], se[1:8]):
    print(f"  alpha_{el:<3s}              = {alpha:+7.1f} +/- {s:5.1f} MPa")
print(f"  k_HP                  = {m3.coef_[7]:+.1f} +/- {se[8]:.1f} MPa.um^(1/2)")
print(f"  R^2 train = {r2_m3:.4f}, R^2 LOO = {r2_loo_m3:.4f}")
print(f"  --> wrote {RESULTS_DIR / 'm3_coefficients.csv'}")
print(f"  --> wrote {RESULTS_DIR / 'm3_model.pkl'}")

# ============================================================
# Fit HV baseline Hall-Petch: HV = H_0 + k_H * d^(-1/2)
# (paper §4.11 reports H_0 = 86.7 HV, k_H = 306 HV.um^(1/2))
# ============================================================
hv = df_hv['HV'].values
d_inv_sqrt_hv = df_hv['d_inv_sqrt'].values

hv_model = LinearRegression().fit(d_inv_sqrt_hv.reshape(-1, 1), hv)
H0 = hv_model.intercept_
kH = hv_model.coef_[0]
r2_hv = hv_model.score(d_inv_sqrt_hv.reshape(-1, 1), hv)

# Optimized exponent for HV
def neg_r2_hv(n):
    feat = df_hv['GrainSize'].values ** (-n)
    m = LinearRegression().fit(feat.reshape(-1, 1), hv)
    return -m.score(feat.reshape(-1, 1), hv)

n_opt_hv = minimize_scalar(neg_r2_hv, bounds=(0.01, 2.5), method='bounded').x

# LOO R^2
loo = LeaveOneOut()
preds_loo_hv = np.zeros(len(hv))
for tr, te in loo.split(d_inv_sqrt_hv):
    mi = LinearRegression().fit(d_inv_sqrt_hv[tr].reshape(-1, 1), hv[tr])
    preds_loo_hv[te] = mi.predict(d_inv_sqrt_hv[te].reshape(-1, 1))
r2_loo_hv = r2_score(hv, preds_loo_hv)

hv_rows = [
    {'parameter': 'H_0',         'value': H0,      'units': 'HV'},
    {'parameter': 'k_H',         'value': kH,      'units': 'HV.um^(1/2)'},
    {'parameter': 'k_H_MPa',     'value': kH * 9.807, 'units': 'MPa.um^(1/2)'},
    {'parameter': 'R2_train',    'value': r2_hv,   'units': ''},
    {'parameter': 'R2_LOO',      'value': r2_loo_hv, 'units': ''},
    {'parameter': 'n_optimized', 'value': n_opt_hv, 'units': '(exponent in d^(-n))'},
    {'parameter': 'n_samples',   'value': len(hv), 'units': ''},
]
hv_df = pd.DataFrame(hv_rows)
hv_df.to_csv(RESULTS_DIR / 'hv_baseline_coefficients.csv', index=False)
dump(hv_model, RESULTS_DIR / 'hv_baseline_model.pkl')

print(f"\nHV Hall-Petch baseline:")
print(f"  H_0       = {H0:+.1f} HV")
print(f"  k_H       = {kH:+.1f} HV.um^(1/2)  ({kH*9.807:.0f} MPa.um^(1/2))")
print(f"  R^2 train = {r2_hv:.4f}, R^2 LOO = {r2_loo_hv:.4f}")
print(f"  n_optimized = {n_opt_hv:.3f}")
print(f"  --> wrote {RESULTS_DIR / 'hv_baseline_coefficients.csv'}")
print(f"  --> wrote {RESULTS_DIR / 'hv_baseline_model.pkl'}")

print(f"\nDone. All artifacts in {RESULTS_DIR}/")
