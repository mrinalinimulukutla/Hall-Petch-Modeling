#!/usr/bin/env python3
"""
Single-panel parity plot: Jiang Direct vs Jiang Recalibrated vs SISSO.

Collapses the previous 3-panel fig10_jiang_comparison.png into one panel
where each alloy contributes one marker per model. Color and marker
shape distinguish models.

Constants copied verbatim from sisso_analysis.py to ensure numerical
agreement with paper's reported R^2 values.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
OUT = f'{PAPER_FIG_DIR}/fig10_jiang_comparison.png'

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# --- Elemental property tables (verbatim from sisso_analysis.py) ---
RADII     = {'Al': 143, 'Co': 125, 'Cr': 128, 'Cu': 128,
             'Fe': 126, 'Mn': 127, 'Ni': 124, 'V': 134}      # pm
VEC_VALS  = {'Al': 3, 'Co': 9, 'Cr': 6, 'Cu': 11,
             'Fe': 8, 'Mn': 7, 'Ni': 10, 'V': 5}
EN        = {'Al': 1.61, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.90,
             'Fe': 1.83, 'Mn': 1.55, 'Ni': 1.91, 'V': 1.63}
SHEAR_MOD = {'Al': 26, 'Co': 75, 'Cr': 115, 'Cu': 48,
             'Fe': 82, 'Mn': 79, 'Ni': 76, 'V': 47}           # GPa
BULK_MOD  = {'Al': 76, 'Co': 180, 'Cr': 160, 'Cu': 140,
             'Fe': 170, 'Mn': 120, 'Ni': 180, 'V': 158}       # GPa
W_coh     = {'Al': 3.39, 'Co': 4.39, 'Cr': 4.10, 'Cu': 3.49,
             'Fe': 4.28, 'Mn': 2.92, 'Ni': 4.44, 'V': 5.31}   # eV/atom
LT_coef   = {'Al': 23.1, 'Co': 13.0, 'Cr': 4.9, 'Cu': 16.5,
             'Fe': 11.8, 'Mn': 21.7, 'Ni': 13.4, 'V': 8.4}    # 10^-6/K
GAMMA_GB  = {'Al': 0.43, 'Co': 0.87, 'Cr': 0.72, 'Cu': 0.60,
             'Fe': 0.78, 'Mn': 0.60, 'Ni': 0.72, 'V': 0.65}   # J/m^2

E_YOUNG = {el: 9 * BULK_MOD[el] * SHEAR_MOD[el]
              / (3 * BULK_MOD[el] + SHEAR_MOD[el])
           for el in ELEMENTS}
S_VED   = {el: (RADII[el] / 100) / VEC_VALS[el] ** (1/3) for el in ELEMENTS}

# --- Load data ---
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df_ys = df.dropna(subset=['YS']).copy()
y = df_ys['YS'].values
d_inv_sqrt = df_ys['d_inv_sqrt'].values
n = len(y)
print(f"Loaded {n} alloys")

# --- Jiang Direct (a=79, b=1.2) ---
jiang_sigma0 = np.zeros(n); jiang_ky = np.zeros(n); jiang_direct = np.zeros(n)
for i, (_, row) in enumerate(df_ys.iterrows()):
    fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
    W_mix  = sum(c * W_coh[el]    for el, c in fracs.items())
    lt_mix = sum(c * LT_coef[el]  for el, c in fracs.items())
    E_mix  = sum(c * E_YOUNG[el]  for el, c in fracs.items())
    g_mix  = sum(c * GAMMA_GB[el] for el, c in fracs.items())
    S_mix  = sum(c * S_VED[el]    for el, c in fracs.items())
    jiang_sigma0[i] = W_mix / (S_mix ** 3 * np.sqrt(lt_mix))
    jiang_ky[i]     = np.sqrt(g_mix * E_mix / lt_mix)
    jiang_direct[i] = 79 * jiang_sigma0[i] + 1.2 * jiang_ky[i] * d_inv_sqrt[i]

# --- Jiang Recalibrated, LOO ---
X_jiang = np.column_stack([jiang_sigma0, jiang_ky * d_inv_sqrt])
preds_jiang_recal = np.zeros(n)
for tr, te in LeaveOneOut().split(X_jiang):
    m = LinearRegression().fit(X_jiang[tr], y[tr])
    preds_jiang_recal[te] = m.predict(X_jiang[te])

r2_jd = r2_score(y, jiang_direct)
r2_jr = r2_score(y, preds_jiang_recal)
print(f"Jiang Direct       R^2 = {r2_jd:+.3f}")
print(f"Jiang Recalib LOO  R^2 = {r2_jr:.3f}")

# --- SISSO Eq. 4 prediction (training-set fit using published coefficients) ---
def oliynyk_terms(row):
    fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
    active = {el: c for el, c in fracs.items() if c > 0}
    cs    = np.array([fracs[el] for el in ELEMENTS])
    rvals = np.array([RADII[el] for el in ELEMENTS])
    evals = np.array([EN[el]    for el in ELEMENTS])
    r_mean   = float(np.sum(cs * rvals))
    r_var    = float(np.sum(cs * (rvals - r_mean) ** 2))
    r_range  = max(RADII[el] for el in active) - min(RADII[el] for el in active)
    EN_mean  = float(np.sum(cs * evals))
    EN_var   = float(np.sum(cs * (evals - EN_mean) ** 2))
    return r_var, r_range, EN_var

sisso_pred = np.zeros(n)
for i, (_, row) in enumerate(df_ys.iterrows()):
    r_var, r_range, EN_var = oliynyk_terms(row)
    dS_mix   = row['dS_mix']
    delta_mu = row['delta_mu']
    sisso_pred[i] = (
        120.4515098926 * (r_var / r_range)
        + 9356.2556899591 * (d_inv_sqrt[i] / dS_mix)
        + 1133.7096001857 * (EN_var / delta_mu)
        - 43.3
    )
r2_sisso_train = r2_score(y, sisso_pred)
print(f"SISSO Eq. 4 train  R^2 = {r2_sisso_train:.3f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(5.2, 4.9))
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
})

lims = [100, 600]
ax.plot(lims, lims, 'k--', linewidth=1.2, alpha=0.5, zorder=1)

ax.scatter(y, jiang_direct, marker='v', s=45, c='#d62728', alpha=0.55,
           edgecolor='black', linewidth=0.4, zorder=2,
           label=f'Jiang Direct ($R^2 = {r2_jd:+.2f}$)')
ax.scatter(y, preds_jiang_recal, marker='s', s=45, c='#ff7f0e', alpha=0.65,
           edgecolor='black', linewidth=0.4, zorder=3,
           label=f'Jiang Recalib (LOO, $R^2 = {r2_jr:.2f}$)')
ax.scatter(y, sisso_pred, marker='o', s=45, c='#2ca02c', alpha=0.75,
           edgecolor='black', linewidth=0.4, zorder=4,
           label=f'SISSO Eq. 4 (train $R^2 = {r2_sisso_train:.2f}$)')

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Experimental YS (MPa)')
ax.set_ylabel('Predicted YS (MPa)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', framealpha=0.92)
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"\nSaved {OUT}")
