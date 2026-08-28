#!/usr/bin/env python3
"""
Variance Floor / R² Ceiling Analysis
=====================================
Decomposes the YS variance into:
  - Between-composition variance (predictable by any composition-only model)
  - Within-composition variance  (irreducible noise floor at fixed comp)
  - Reported per-alloy SD_YS²    (measurement-noise floor)

Yields three R² ceilings:
  R²_comp_ceiling        = 1 - var(within_composition) / var(total)
  R²_noise_ceiling       = 1 - <SD_YS²> / var(total)
  R²_comp_x_d_ceiling    = 1 - var(within_composition_x_grain_bin) / var(total)

Use these as denominators when reporting how close each model is to the
data's intrinsic limit (the pre-modeling diagnostics framing).

Outputs
-------
  results/variance_ceiling.csv
  analysis_plots/82_variance_ceiling.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# ============================================================
# 1. LOAD
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv').dropna(subset=['YS']).reset_index(drop=True)
print(f"n = {len(df)}, var(YS) = {df['YS'].var(ddof=1):.1f}")

# ============================================================
# 2. COMPOSITION REPLICATE GROUPS
# ============================================================
comp_cols = [f'{e}_frac' for e in ELEMENTS]
# Round composition to mitigate floating-point microvariation between rows
comp_key = df[comp_cols].round(4).astype(str).agg('|'.join, axis=1)
df['_comp_key'] = comp_key

groups = df.groupby('_comp_key')
sizes = groups.size()
print(f"\nReplicate groups: {len(sizes)} unique compositions")
print(f"  singletons:        {(sizes == 1).sum()}")
print(f"  >= 2 alloys:       {(sizes >= 2).sum()} "
      f"({(sizes[sizes >= 2]).sum()} total alloys)")

# ============================================================
# 3. WITHIN-COMPOSITION VARIANCE
# ============================================================
within_terms = []
for key, sub in groups:
    if len(sub) >= 2:
        within_terms.append(sub['YS'].var(ddof=1) * (len(sub) - 1))
ss_within = float(np.sum(within_terms))
df_within = int(sum(len(sub) - 1 for _, sub in groups if len(sub) >= 2))
var_within = ss_within / df_within if df_within > 0 else np.nan
print(f"\nWithin-composition var(YS) = {var_within:.1f} "
      f"(pooled over {df_within} d.o.f.)")

# ============================================================
# 4. CEILINGS
# ============================================================
var_total = df['YS'].var(ddof=1)
r2_comp_ceiling = 1.0 - var_within / var_total if not np.isnan(var_within) else np.nan

# Per-alloy SD_YS (if present in raw data)
sd_col = next((c for c in ('SD_YS', 'YS_SD') if c in df.columns), None)
if sd_col is not None:
    sd_vals = pd.to_numeric(df[sd_col], errors='coerce').dropna()
    mean_sd2 = float((sd_vals ** 2).mean())
    r2_noise_ceiling = 1.0 - mean_sd2 / var_total
    print(f"\n<SD_YS²> = {mean_sd2:.1f} (over {len(sd_vals)} alloys)")
else:
    mean_sd2 = np.nan
    r2_noise_ceiling = np.nan
    print("[warn] no SD_YS column found — noise ceiling unavailable.")

print(f"\nCEILINGS")
print(f"  R²_composition_ceiling = {r2_comp_ceiling:.3f}"
      f"  (best a composition-only model can do)")
print(f"  R²_noise_ceiling       = {r2_noise_ceiling:.3f}"
      f"  (limit set by per-alloy measurement noise)")

# ============================================================
# 5. SAVE
# ============================================================
out = pd.DataFrame([{
    'n_alloys': len(df),
    'n_unique_comp': int(len(sizes)),
    'n_singletons': int((sizes == 1).sum()),
    'n_replicated_groups': int((sizes >= 2).sum()),
    'var_total_YS': var_total,
    'var_within_composition': var_within,
    'mean_SD_YS_sq': mean_sd2,
    'R2_composition_ceiling': r2_comp_ceiling,
    'R2_noise_ceiling': r2_noise_ceiling,
}])
out.to_csv(f'{RESULTS_DIR}/variance_ceiling.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/variance_ceiling.csv")

# ============================================================
# 6. PLOT — variance decomposition bars
# ============================================================
labels = ['Total', 'Within-composition', '<SD_YS²>']
vals = [var_total,
        var_within if not np.isnan(var_within) else 0,
        mean_sd2  if not np.isnan(mean_sd2) else 0]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(labels, vals, color=['#444', '#888', '#bbb'], edgecolor='black')
ax.set_ylabel('var(YS)  [MPa²]')
ax.set_title('YS variance decomposition: ceilings for any model')
for lbl, v in zip(labels, vals):
    if v > 0:
        ax.text(lbl, v, f'{v:.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/82_variance_ceiling.png', dpi=150)
plt.close()
print(f"Wrote {PLOTS_DIR}/82_variance_ceiling.png")
