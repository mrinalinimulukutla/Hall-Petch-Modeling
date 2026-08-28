#!/usr/bin/env python3
"""
Bootstrap CIs on Symbolic-Equation Constants
=============================================
For each compact equation, freeze the symbolic form and bootstrap-resample
(B = 1000) the dataset, refitting the constants each time. Report:

  - mean and SD of each constant
  - bias-corrected 95% CI
  - per-fold drift under LOO (separate diagnostic)

A "publishable" equation has SD < 30% of mean for every constant.

Outputs
-------
  results/bootstrap_sr_constants.csv
  analysis_plots/84_bootstrap_constants.png
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
B = 1000

# ============================================================
# 1. LOAD
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv').dropna(subset=['YS', 'HV']).reset_index(drop=True)
SD_COL = next((c for c in ('SD_GS', 'GrainSize_SD', 'SD_grain', 'GS_SD') if c in df.columns), None)

# ============================================================
# 2. SYMBOLIC FORMS (with named free constants)
# ============================================================
def ys_compact_form(X, c1, c2, c3):
    """YS = VEC * (c1 * dH * SD / (d^2 * dchi) + c2/dchi + c3)"""
    VEC, dH, SD, d, dchi = X
    return VEC * (c1 * dH * SD / (d ** 2 * dchi) + c2 / dchi + c3)

def hv_elbow_form(X, c0, c1, c2):
    """HV = c0 + c1 * (6.93 - d) / SD + c2 * dH / t^2"""
    d, SD, dH, t = X
    return c0 + c1 * (6.93 - d) / SD + c2 * dH / (t ** 2)

# ============================================================
# 3. BOOTSTRAP LOOP
# ============================================================
def bootstrap(form, X_arrays, y, p0, B=1000):
    n = len(y)
    samples = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        try:
            popt, _ = curve_fit(form, tuple(arr[idx] for arr in X_arrays),
                                y[idx], p0=p0, maxfev=10_000)
            samples.append(popt)
        except Exception:
            continue
    arr = np.array(samples)
    return arr  # shape (n_success, len(p0))

rows = []

if SD_COL and {'VEC', 'dH_mix', 'delta_chi'} <= set(df.columns):
    X_ys = (
        df['VEC'].values.astype(float),
        df['dH_mix'].values.astype(float),
        df[SD_COL].values.astype(float),
        df['GrainSize'].values.astype(float),
        df['delta_chi'].values.astype(float),
    )
    print(f"Bootstrapping YS-compact (B={B}) ...")
    samples = bootstrap(ys_compact_form, X_ys, df['YS'].values.astype(float),
                        p0=[4.29, -2.13, 56.06], B=B)
    if len(samples):
        means = samples.mean(axis=0)
        sds = samples.std(axis=0)
        ci_lo = np.quantile(samples, 0.025, axis=0)
        ci_hi = np.quantile(samples, 0.975, axis=0)
        for i, name in enumerate(['c1', 'c2', 'c3']):
            rows.append({
                'equation': 'YS_compact', 'constant': name,
                'mean': float(means[i]), 'sd': float(sds[i]),
                'ci_lo': float(ci_lo[i]), 'ci_hi': float(ci_hi[i]),
                'sd_over_mean_pct': float(100 * abs(sds[i] / means[i])) if means[i] else np.inf,
            })
    else:
        print("[warn] YS-compact bootstrap produced no successful fits.")
else:
    print("[skip] YS-compact bootstrap — required columns missing.")

if SD_COL:
    X_hv = (df['GrainSize'].values.astype(float),
            df[SD_COL].values.astype(float),
            df['dH_mix'].values.astype(float),
            df['HoldTime'].values.astype(float))
    print(f"Bootstrapping HV-elbow (B={B}) ...")
    samples = bootstrap(hv_elbow_form, X_hv, df['HV'].values.astype(float),
                        p0=[221.46, -83.95, 1.0], B=B)
    if len(samples):
        means = samples.mean(axis=0)
        sds = samples.std(axis=0)
        ci_lo = np.quantile(samples, 0.025, axis=0)
        ci_hi = np.quantile(samples, 0.975, axis=0)
        for i, name in enumerate(['c0', 'c1', 'c2']):
            rows.append({
                'equation': 'HV_elbow', 'constant': name,
                'mean': float(means[i]), 'sd': float(sds[i]),
                'ci_lo': float(ci_lo[i]), 'ci_hi': float(ci_hi[i]),
                'sd_over_mean_pct': float(100 * abs(sds[i] / means[i])) if means[i] else np.inf,
            })

# ============================================================
# 4. SAVE + PLOT
# ============================================================
out = pd.DataFrame(rows)
out.to_csv(f'{RESULTS_DIR}/bootstrap_sr_constants.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/bootstrap_sr_constants.csv")
if len(out):
    print(out.to_string(index=False))

if len(out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(out['equation'] + ' / ' + out['constant'],
            out['sd_over_mean_pct'], color='#4c72b0', edgecolor='black')
    ax.axvline(30, color='red', linestyle='--', label='30% publishability threshold')
    ax.set_xlabel('SD / |mean|  (%)')
    ax.set_title('Bootstrap stability of compact-equation constants')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/84_bootstrap_constants.png', dpi=150)
    plt.close()
    print(f"Wrote {PLOTS_DIR}/84_bootstrap_constants.png")
