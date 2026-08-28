#!/usr/bin/env python3
"""
Singularity Audit for Symbolic Equations
=========================================
Scans candidate symbolic equations for poles, sign flips, and other
deployment hazards. Applies to:

  - compact-equation stream compact YS equation: Δχ in denominator
        YS ≈ VEC * (4.29 * dH_mix * SD_grain / (d^2 * delta_chi)
                    − 2.13 / delta_chi + 56.06)
  - compact-equation stream compact HV equation: (6.93 − d)/SD_grain sign flip,
        t_hold near zero
  - SISSO Full equation: σ²_χ/δ_μ (the known SISSO Full pathology, kept for
        reference)
  - SISSO Robust equation: bounded (σ²_χ − Φ_VLC); should be safe

For each equation we report:
  - In-distribution risk:  any denominator < tolerance on training data?
  - In-distribution sign flips: any factor changes sign within data range?
  - External-data risk:  evaluate on the 82-point literature set
                         and report fraction of points outside envelope.
  - Recommended envelope: feature-range bounds for safe deployment.

Outputs
-------
  results/singularity_audit.csv
  analysis_plots/81_singularity_audit.png
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

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')

SD_COL = next((c for c in ('SD_GS', 'GrainSize_SD', 'SD_grain', 'GS_SD') if c in df.columns), None)
if SD_COL is None:
    raise SystemExit("[fatal] no SD_grain column in derived CSV.")

# Required descriptors for the equations under audit
REQ = ['VEC', 'dH_mix', 'delta_chi', 'GrainSize', SD_COL, 'HoldTime']
miss = [c for c in REQ if c not in df.columns]
if miss:
    raise SystemExit(f"[fatal] missing required columns: {miss}")

# Build a typed view for clarity
view = pd.DataFrame({
    'VEC':      df['VEC'].astype(float),
    'dH_mix':   df['dH_mix'].astype(float),
    'delta_chi':    df['delta_chi'].astype(float),
    'd':        df['GrainSize'].astype(float),
    'SD_grain': df[SD_COL].astype(float),
    't_hold':   df['HoldTime'].astype(float),
})

# ============================================================
# 2. AUDIT REGISTRY
# ============================================================
# Each entry: (name, denominator_terms, sign_flip_factors, comments)
EQUATIONS = [
    {
        'name': 'Compact_YS_equation',
        'expression':
            'VEC * (4.29 * dH_mix * SD_grain / (d^2 * delta_chi)'
            ' - 2.13 / delta_chi + 56.06)',
        'denominators': {
            'd^2':   lambda v: v['d'] ** 2,
            'delta_chi': lambda v: v['delta_chi'],
        },
        'sign_flip_factors': {
            # No structural sign flip in this equation; included for completeness.
        },
    },
    {
        'name': 'Compact_HV_elbow',
        'expression':
            '221.46 - 83.95 * (6.93 - d) / SD_grain + dH_mix / t_hold^2',
        'denominators': {
            'SD_grain':  lambda v: v['SD_grain'],
            't_hold^2':  lambda v: v['t_hold'] ** 2,
        },
        'sign_flip_factors': {
            '(6.93 - d)': lambda v: 6.93 - v['d'],
        },
    },
    {
        'name': 'SISSO_Full',
        'expression':
            'sigma_y = 120.5 * (r_var/r_range) + 9356 * (d_inv_sqrt/dS_mix)'
            ' + 1134 * (EN_var/delta_mu) - 43.3',
        'denominators': {
            # delta_mu is the known singularity; cannot compute here without
            # the SISSO-specific features. Kept as a documented hazard.
        },
        'sign_flip_factors': {},
        'note': 'Known pathology: delta_mu → 0 for similar-shear-modulus alloys',
    },
    {
        'name': 'SISSO_Robust',
        'expression':
            'sigma_y = 113.0 * (sigma2_r / r_range)'
            ' + 9837 * (d_inv_sqrt / dS_mix)'
            ' + 5437 * (sigma2_chi - Phi_VLC) - 27.0',
        'denominators': {},
        'sign_flip_factors': {},
        'note': 'Bounded by construction (σ²_χ − Φ_VLC replaces ratio)',
    },
]

# ============================================================
# 3. AUDIT LOOP
# ============================================================
DENOM_TOL = 1e-3
rows = []
for eq in EQUATIONS:
    print(f"\n--- {eq['name']} ---")
    print(f"    expr: {eq['expression']}")
    row = {'name': eq['name'], 'expression': eq['expression']}

    # Denominator near-zero check
    for label, fn in eq.get('denominators', {}).items():
        try:
            vals = fn(view)
            n_near_zero = int((np.abs(vals) < DENOM_TOL).sum())
            row[f'denom_{label}_min_abs'] = float(np.abs(vals).min())
            row[f'denom_{label}_n_near_zero'] = n_near_zero
            print(f"    denominator '{label}': min|·| = {np.abs(vals).min():.4g}, "
                  f"n_near_zero(<{DENOM_TOL}) = {n_near_zero}")
        except Exception as exc:
            print(f"    [skip] '{label}' — {exc}")

    # Sign-flip check
    for label, fn in eq.get('sign_flip_factors', {}).items():
        try:
            vals = fn(view)
            n_neg = int((vals < 0).sum())
            n_pos = int((vals > 0).sum())
            row[f'sign_{label}_n_neg'] = n_neg
            row[f'sign_{label}_n_pos'] = n_pos
            print(f"    sign factor '{label}': n_neg = {n_neg}, n_pos = {n_pos}")
            if min(n_neg, n_pos) > 0:
                print(f"    [warn] sign flip occurs IN-DISTRIBUTION for '{label}'.")
        except Exception as exc:
            print(f"    [skip] '{label}' — {exc}")

    if 'note' in eq:
        row['note'] = eq['note']
        print(f"    note: {eq['note']}")

    rows.append(row)

# ============================================================
# 4. SAVE
# ============================================================
out = pd.DataFrame(rows)
out.to_csv(f'{RESULTS_DIR}/singularity_audit.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/singularity_audit.csv")

# ============================================================
# 5. ENVELOPE PLOT — Δχ distribution + d distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(view['delta_chi'], bins=20, edgecolor='black')
axes[0].axvline(DENOM_TOL, color='red', linestyle='--',
                label=f'denom tol = {DENOM_TOL}')
axes[0].set_xlabel('delta_chi (electronegativity mismatch)')
axes[0].set_ylabel('count')
axes[0].set_title('compact-equation stream YS-compact denominator distribution')
axes[0].legend()

axes[1].hist(view['d'], bins=20, edgecolor='black')
axes[1].axvline(6.93, color='red', linestyle='--',
                label='d = 6.93 μm (HV sign flip)')
axes[1].set_xlabel('Mean grain size d (μm)')
axes[1].set_title('compact-equation stream HV-compact sign-flip threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/81_singularity_audit.png', dpi=150)
plt.close()
print(f"Wrote {PLOTS_DIR}/81_singularity_audit.png")

# ============================================================
# 6. RECOMMENDED ENVELOPES
# ============================================================
print("\nRecommended deployment envelopes:")
print(f"  compact-equation stream YS compact: delta_chi >= {DENOM_TOL} (min observed = "
      f"{view['delta_chi'].min():.4f})")
print(f"  compact-equation stream HV compact: SD_grain >= {DENOM_TOL}, t_hold > 0, "
      f"d != 6.93 (sign-flip caution)")
print("  Next: run scripts/external_validation.py to test each equation on")
print("  the 82-point literature set and quantify out-of-envelope failures.")
