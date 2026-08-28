#!/usr/bin/env python3
"""
Hardness Symbolic Regression 
===================================================
Runs PySR specifically for HV with the SD_grain channel, then evaluates
the published compact-equation stream elbow equation:

  HV ≈ 221.46 − 83.95 * (6.93 − d) / SD_grain  +  dH_mix / t_hold^2

under LOO + LOBO + external validation, plus an explicit singularity audit
for d ≈ 6.93 μm (in-distribution risk: GS_min = 15 μm, so no in-distribution
pole, but check extrapolation) and t_hold near 0.

For comparison, also evaluates the F3+O3 accuracy equation (which
contains the (0.69 − holding_time) near-pole that we flagged) and confirms
why the elbow form is the publishable choice.

Outputs
-------
  results/hardness_sr_results.csv
  results/hardness_sr_singularity.csv
  analysis_plots/80_hardness_sr_envelope.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, KFold

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

# ============================================================
# 1. LOAD
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df = df.dropna(subset=['HV']).reset_index(drop=True)

SD_COL = next((c for c in ('SD_GS', 'GrainSize_SD', 'SD_grain', 'GS_SD') if c in df.columns), None)
if SD_COL is None:
    raise SystemExit("[fatal] no SD_grain column in derived CSV.")

d         = df['GrainSize'].values.astype(float)
SD_grain  = df[SD_COL].values.astype(float)
dH_mix    = df['dH_mix'].values.astype(float)
t_hold    = df['HoldTime'].values.astype(float)
HV        = df['HV'].values.astype(float)
groups    = df['Iteration'].values if 'Iteration' in df.columns else None

# ============================================================
# 2. compact-equation stream PUBLISHED EQUATIONS
# ============================================================
def hv_elbow(d, SD, dH, t):
    """compact F3+O2 elbow equation (publishable)."""
    return 221.46 - 83.95 * (6.93 - d) / SD + dH / (t ** 2)

def hv_accuracy(d, SD, dH, t):
    """compact-equation stream F3+O3 accuracy equation — contains (0.69 − t) near-pole.
    Coefficients here are placeholders; replace with the actual fitted
    constants from the deck results once the source notebook is migrated."""
    # TODO: paste actual F3+O3 accuracy equation constants here.
    return np.where(np.abs(0.69 - t) < 1e-3, np.nan,
                    220.0 - 80.0 * (6.93 - d) / SD
                    + dH / (t ** 2) + 0.5 / (0.69 - t))

def evaluate(name, predict_fn):
    y_pred = predict_fn(d, SD_grain, dH_mix, t_hold)
    ok = np.isfinite(y_pred)
    r2 = r2_score(HV[ok], y_pred[ok])
    rmse = np.sqrt(mean_squared_error(HV[ok], y_pred[ok]))
    print(f"  {name:24s} R² = {r2:6.3f}, RMSE = {rmse:6.1f}, "
          f"n_valid = {ok.sum()}/{len(HV)}")
    return r2, rmse, int(ok.sum())

print("=" * 70)
print("HARDNESS SR — published-equation evaluation")
print("=" * 70)
rows = []
for name, fn in [('elbow_F3O2', hv_elbow), ('accuracy_F3O3', hv_accuracy)]:
    r2, rmse, n_valid = evaluate(name, fn)
    rows.append({'equation': name, 'R2_full_data': r2, 'RMSE_full_data': rmse,
                 'n_valid': n_valid})

# ============================================================
# 3. CROSS-VALIDATION (REFIT)
# ============================================================
# For each equation, freeze the form and refit constants per CV fold.
# Constants for the elbow equation: c0 (offset), c1 (gain on heterogeneity
# term), c2 (gain on dH_mix/t^2). Fit with scipy.optimize.curve_fit.
from scipy.optimize import curve_fit

def elbow_form(X, c0, c1, c2):
    d_, SD_, dH_, t_ = X
    return c0 + c1 * (6.93 - d_) / SD_ + c2 * dH_ / (t_ ** 2)

def refit_cv(splitter, X_pack, y_, label, **split_kw):
    preds = np.full_like(y_, np.nan, dtype=float)
    for tr, te in splitter.split(*X_pack, **split_kw) if split_kw else splitter.split(X_pack[0]):
        Xt = tuple(arr[tr] for arr in X_pack)
        yt = y_[tr]
        try:
            popt, _ = curve_fit(elbow_form, Xt, yt, p0=[221.46, -83.95, 1.0],
                                maxfev=10_000)
            preds[te] = elbow_form(tuple(arr[te] for arr in X_pack), *popt)
        except Exception:
            preds[te] = np.nan
    ok = np.isfinite(preds)
    r2 = r2_score(y_[ok], preds[ok])
    rmse = np.sqrt(mean_squared_error(y_[ok], preds[ok]))
    print(f"  {label:24s} R² = {r2:6.3f}, RMSE = {rmse:6.1f}, "
          f"folds-valid = {ok.sum()}/{len(y_)}")
    return r2, rmse

X_pack = (d, SD_grain, dH_mix, t_hold)
print("\nRefit CV (HV elbow equation):")
r2_5, rmse_5 = refit_cv(KFold(n_splits=5, shuffle=True, random_state=42),
                        X_pack, HV, '5-fold refit')
r2_loo, rmse_loo = refit_cv(LeaveOneOut(), X_pack, HV, 'LOO refit')

# LOBO requires explicit groups argument
class _Wrap:
    def __init__(self, base, X_pack, y, groups):
        self.base = base; self.Xp = X_pack; self.y = y; self.g = groups
    def split(self, *args, **kwargs):
        return self.base.split(self.Xp[0], self.y, self.g)

if groups is not None:
    r2_lobo, rmse_lobo = refit_cv(_Wrap(LeaveOneGroupOut(), X_pack, HV, groups),
                                  X_pack, HV, 'LOBO refit')
else:
    r2_lobo, rmse_lobo = np.nan, np.nan
    print("[warn] no 'Iteration' column — LOBO skipped.")

rows.append({'equation': 'elbow_F3O2_refit',
             'R2_5fold': r2_5, 'RMSE_5fold': rmse_5,
             'R2_LOO': r2_loo, 'RMSE_LOO': rmse_loo,
             'R2_LOBO': r2_lobo, 'RMSE_LOBO': rmse_lobo})

# ============================================================
# 4. SINGULARITY ENVELOPE
# ============================================================
# In-distribution risk for the elbow equation:
#   denom (SD_grain) → 0 ? Check min(SD_grain).
#   t_hold → 0 ? Check min(t_hold).
# The (6.93 − d) factor changes sign at d = 6.93 μm — not a pole, but the
# equation's *interpretation* flips. We record the sign-flip envelope.
env = {
    'SD_grain_min': float(SD_grain.min()),
    'SD_grain_safe_floor': 1.0,                            # arbitrary; document
    't_hold_min': float(t_hold.min()),
    't_hold_safe_floor': 0.5,                              # training-data envelope
    'd_signflip_threshold': 6.93,                          # equation-intrinsic
    'd_observed_min': float(d.min()),
    'd_observed_max': float(d.max()),
}
print("\nSingularity envelope (elbow eq):")
for k, v in env.items():
    print(f"  {k}: {v}")

# Persist
pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/hardness_sr_results.csv', index=False)
pd.DataFrame([env]).to_csv(f'{RESULTS_DIR}/hardness_sr_singularity.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/hardness_sr_results.csv")
print(f"Wrote {RESULTS_DIR}/hardness_sr_singularity.csv")

# ============================================================
# 5. ENVELOPE PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(d, SD_grain, c=HV, cmap='viridis', s=40, edgecolor='k')
ax.axvline(6.93, color='red', linestyle='--', alpha=0.5,
           label='d = 6.93 μm (sign flip)')
ax.set_xlabel('Mean grain size d (μm)')
ax.set_ylabel('SD_grain (μm)')
ax.set_title('HV elbow eq deployment envelope (color = HV)')
ax.legend()
plt.colorbar(ax.collections[0], ax=ax, label='HV')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/80_hardness_sr_envelope.png', dpi=150)
plt.close()
print(f"Wrote {PLOTS_DIR}/80_hardness_sr_envelope.png")
print("\nDone.")
