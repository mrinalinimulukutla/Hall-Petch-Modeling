#!/usr/bin/env python3
"""
Monte Carlo Grain-Size Sensitivity
===================================
Perturbs each alloy's grain size by SD_grain / sqrt(N_grains) (N_grains
defaults to 10 — conservative) and refits the compact YS and HV equations
N_REP = 1000 times. Reports the resulting distribution of every constant.

This is the test for "are our equation constants robust to grain-size
measurement noise?" — a question Acta will absolutely ask.

Outputs
-------
  results/mc_grain_size_sensitivity.csv
  analysis_plots/87_mc_grain_size.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

RNG = np.random.default_rng(seed=42)
N_REP = 1000
N_GRAINS_DEFAULT = 10                # conservative; if EBSD scan had >50, halve perturbation

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv').dropna(subset=['YS', 'HV']).reset_index(drop=True)
SD_COL = next((c for c in ('SD_GS', 'GrainSize_SD', 'SD_grain', 'GS_SD') if c in df.columns), None)
if SD_COL is None:
    raise SystemExit("[fatal] no SD_grain column.")

d0     = df['GrainSize'].values.astype(float)
SD     = df[SD_COL].values.astype(float)
sigma  = SD / np.sqrt(N_GRAINS_DEFAULT)
y_ys   = df['YS'].values.astype(float)
y_hv   = df['HV'].values.astype(float)

# Forms (same as bootstrap_sr_constants.py)
def ys_form(X, c1, c2, c3):
    VEC, dH, SD_, d, dchi = X
    return VEC * (c1 * dH * SD_ / (d ** 2 * dchi) + c2 / dchi + c3)

def hv_form(X, c0, c1, c2):
    d, SD_, dH, t = X
    return c0 + c1 * (6.93 - d) / SD_ + c2 * dH / (t ** 2)

VEC  = df['VEC'].values.astype(float)
dH   = df['dH_mix'].values.astype(float)
dchi = df['delta_chi'].values.astype(float)
t    = df['HoldTime'].values.astype(float)

ys_samples = []
hv_samples = []
for _ in range(N_REP):
    d_pert = d0 + RNG.normal(0, sigma)
    d_pert = np.maximum(d_pert, 1.0)            # guard against negatives
    d_inv_sqrt_pert = d_pert ** -0.5
    try:
        popt, _ = curve_fit(ys_form, (VEC, dH, SD, d_pert, dchi), y_ys,
                            p0=[4.29, -2.13, 56.06], maxfev=10_000)
        ys_samples.append(popt)
    except Exception:
        pass
    try:
        popt, _ = curve_fit(hv_form, (d_pert, SD, dH, t), y_hv,
                            p0=[221.46, -83.95, 1.0], maxfev=10_000)
        hv_samples.append(popt)
    except Exception:
        pass

def summarize(name, arr, names):
    arr = np.asarray(arr)
    out = []
    for i, nm in enumerate(names):
        out.append({
            'equation': name, 'constant': nm,
            'mean': float(arr[:, i].mean()),
            'sd':   float(arr[:, i].std()),
            'ci_lo': float(np.quantile(arr[:, i], 0.025)),
            'ci_hi': float(np.quantile(arr[:, i], 0.975)),
            'sd_over_mean_pct': float(100 * abs(arr[:, i].std() / arr[:, i].mean()))
                if arr[:, i].mean() else np.inf,
        })
    return out

rows = []
if len(ys_samples):
    rows += summarize('YS_compact', ys_samples, ['c1', 'c2', 'c3'])
if len(hv_samples):
    rows += summarize('HV_elbow', hv_samples, ['c0', 'c1', 'c2'])

out = pd.DataFrame(rows)
out.to_csv(f'{RESULTS_DIR}/mc_grain_size_sensitivity.csv', index=False)
print(f"Wrote {RESULTS_DIR}/mc_grain_size_sensitivity.csv")
print(out.to_string(index=False))

if len(out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(out['equation'] + ' / ' + out['constant'],
            out['sd_over_mean_pct'], color='#dd8452', edgecolor='black')
    ax.axvline(30, color='red', linestyle='--')
    ax.set_xlabel('SD / |mean|  (%)  under MC GS perturbation')
    ax.set_title(f'MC grain-size sensitivity (N_grains = {N_GRAINS_DEFAULT}, N_rep = {N_REP})')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/87_mc_grain_size.png', dpi=150)
    plt.close()
    print(f"Wrote {PLOTS_DIR}/87_mc_grain_size.png")
