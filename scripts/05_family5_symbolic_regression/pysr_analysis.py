#!/usr/bin/env python3
"""
PySR Symbolic Regression for HEA Strengthening
================================================
Discovers interpretable analytical equations:
  σ_y = σ_0(comp) + k_HP(comp) · d^(-1/2)

Uses PySR (Julia-backed symbolic regression) to search for
composition-dependent strengthening laws.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

print("=" * 70)
print("PySR SYMBOLIC REGRESSION ANALYSIS")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df_ys = df.dropna(subset=['YS'])
print(f"Loaded {len(df_ys)} alloys with YS data")

# ============================================================
# PREPARE FEATURES
# ============================================================
# Strategy 1: Full symbolic regression on all features
# Use atomic fractions + d^(-1/2) + processing
feature_cols = [f'{el}_frac' for el in ELEMENTS] + ['d_inv_sqrt', 'ColdWork', 'RecrystT']

X = df_ys[feature_cols].values
y = df_ys['YS'].values
feature_names = feature_cols

print(f"\nFeatures ({len(feature_names)}): {feature_names}")
print(f"Samples: {len(y)}")
print(f"YS range: {y.min():.0f} - {y.max():.0f} MPa")

# ============================================================
# PySR SYMBOLIC REGRESSION
# ============================================================
print("\nInitializing PySR (first run installs Julia dependencies)...")

from pysr import PySRRegressor

# Strategy 1: Full regression — discover σ_y = f(comp, d, process)
model_full = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    populations=20,
    population_size=40,
    maxsize=25,
    maxdepth=6,
    parsimony=0.005,
    nested_constraints={
        "sqrt": {"sqrt": 0},
        "square": {"square": 0},
    },
    model_selection="best",
    temp_equation_file=True,
    tempdir=f'{BASE}/pysr_temp',
    verbosity=1,
    progress=False,
    random_state=42,
    deterministic=True,
    parallelism='serial',
    turbo=False,
    batching=False,
)

print("\nRunning PySR full regression (σ_y as function of all features)...")
model_full.fit(X, y, variable_names=feature_names)

print("\n" + "=" * 70)
print("PySR RESULTS: Full Model")
print("=" * 70)
print(model_full)

# Get Pareto front of equations
equations = model_full.equations_
print(f"\n  Pareto front ({len(equations)} equations):")
print(f"  {'Complexity':>10s} {'Loss':>12s} {'Score':>10s}  Equation")
print(f"  {'-' * 80}")
for _, eq in equations.iterrows():
    print(f"  {eq['complexity']:>10d} {eq['loss']:>12.4f} {eq['score']:>10.6f}  {eq['equation']}")

# Best equation
best_eq = model_full.get_best()
print(f"\n  Best equation: {model_full.sympy()}")
y_pred_full = model_full.predict(X)
r2_full = r2_score(y, y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
print(f"  R² = {r2_full:.4f}, RMSE = {rmse_full:.2f} MPa")

# ============================================================
# Strategy 2: Discover composition-dependent σ_0
# Residuals after removing Hall-Petch: σ_0 = YS - k_HP·d^(-1/2)
# ============================================================
print("\n" + "=" * 70)
print("PySR: Discovering composition-dependent σ_0")
print("=" * 70)

from sklearn.linear_model import LinearRegression

# First get k_HP from simple Hall-Petch
reg_hp = LinearRegression().fit(df_ys[['d_inv_sqrt']].values, y)
k_HP = reg_hp.coef_[0]
sigma_0_avg = reg_hp.intercept_

# Residuals = σ_0(comp) ≈ YS - k_HP·d^(-1/2)
sigma_0_residuals = y - k_HP * df_ys['d_inv_sqrt'].values

# Features: just compositions
X_comp = df_ys[[f'{el}_frac' for el in ELEMENTS]].values
comp_names = [f'{el}_frac' for el in ELEMENTS]

model_sigma0 = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    populations=15,
    population_size=33,
    maxsize=20,
    maxdepth=5,
    parsimony=0.005,
    model_selection="best",
    temp_equation_file=True,
    tempdir=f'{BASE}/pysr_temp_sigma0',
    verbosity=1,
    progress=False,
    random_state=42,
    deterministic=True,
    parallelism='serial',
    turbo=False,
    batching=False,
)

print(f"\nFitting σ_0(comp) = YS - {k_HP:.0f}·d^(-1/2)...")
model_sigma0.fit(X_comp, sigma_0_residuals, variable_names=comp_names)

print("\n  σ_0(comp) Pareto front:")
equations_s0 = model_sigma0.equations_
for _, eq in equations_s0.iterrows():
    print(f"    complexity={eq['complexity']:2d}  loss={eq['loss']:8.2f}  {eq['equation']}")

print(f"\n  Best σ_0(comp): {model_sigma0.sympy()}")
s0_pred = model_sigma0.predict(X_comp)
r2_s0 = r2_score(sigma_0_residuals, s0_pred)
print(f"  R² (σ_0 fit) = {r2_s0:.4f}")

# ============================================================
# Strategy 3: Discover composition-dependent k_HP
# For each alloy, estimate k_HP = (YS - σ_0_avg) / d^(-1/2)
# ============================================================
print("\n" + "=" * 70)
print("PySR: Discovering composition-dependent k_HP")
print("=" * 70)

# Use per-alloy residual: k_HP_eff = (YS - σ_0_avg) / d^(-1/2)
k_HP_effective = (y - sigma_0_avg) / df_ys['d_inv_sqrt'].values

model_kHP = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    populations=15,
    population_size=33,
    maxsize=20,
    maxdepth=5,
    parsimony=0.005,
    model_selection="best",
    temp_equation_file=True,
    tempdir=f'{BASE}/pysr_temp_kHP',
    verbosity=1,
    progress=False,
    random_state=42,
    deterministic=True,
    parallelism='serial',
    turbo=False,
    batching=False,
)

print(f"\nFitting k_HP(comp)...")
model_kHP.fit(X_comp, k_HP_effective, variable_names=comp_names)

print("\n  k_HP(comp) Pareto front:")
equations_kHP = model_kHP.equations_
for _, eq in equations_kHP.iterrows():
    print(f"    complexity={eq['complexity']:2d}  loss={eq['loss']:10.2f}  {eq['equation']}")

print(f"\n  Best k_HP(comp): {model_kHP.sympy()}")
kHP_pred = model_kHP.predict(X_comp)
r2_kHP = r2_score(k_HP_effective, kHP_pred)
print(f"  R² (k_HP fit) = {r2_kHP:.4f}")

# ============================================================
# COMBINED EQUATION: σ_y = σ_0(comp) + k_HP(comp)·d^(-1/2)
# ============================================================
print("\n" + "=" * 70)
print("COMBINED EQUATION PERFORMANCE")
print("=" * 70)

ys_combined = model_sigma0.predict(X_comp) + model_kHP.predict(X_comp) * df_ys['d_inv_sqrt'].values
r2_combined = r2_score(y, ys_combined)
rmse_combined = np.sqrt(mean_squared_error(y, ys_combined))
print(f"\n  σ_y = σ_0(comp) + k_HP(comp)·d^(-1/2)")
print(f"  σ_0(comp) = {model_sigma0.sympy()}")
print(f"  k_HP(comp) = {model_kHP.sympy()}")
print(f"  R² = {r2_combined:.4f}, RMSE = {rmse_combined:.2f} MPa")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# (a) Full model parity
ax = axes[0, 0]
ax.scatter(y, y_pred_full, c='steelblue', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
lims = [min(y.min(), y_pred_full.min()) * 0.9, max(y.max(), y_pred_full.max()) * 1.1]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Experimental YS (MPa)', fontsize=12)
ax.set_ylabel('PySR Predicted YS (MPa)', fontsize=12)
ax.set_title(f'PySR Full Model: R²={r2_full:.3f}', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# (b) σ_0(comp) fit
ax = axes[0, 1]
ax.scatter(sigma_0_residuals, s0_pred, c='coral', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
lims_s0 = [min(sigma_0_residuals.min(), s0_pred.min()) * 0.9,
           max(sigma_0_residuals.max(), s0_pred.max()) * 1.1]
ax.plot(lims_s0, lims_s0, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Actual σ₀ (YS − k·d⁻¹/²) [MPa]', fontsize=12)
ax.set_ylabel('PySR σ₀(comp) [MPa]', fontsize=12)
ax.set_title(f'σ₀(comp) Discovery: R²={r2_s0:.3f}', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# (c) k_HP(comp) fit
ax = axes[1, 0]
ax.scatter(k_HP_effective, kHP_pred, c='green', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
ax.set_xlabel('Effective k_HP [MPa·µm^(1/2)]', fontsize=12)
ax.set_ylabel('PySR k_HP(comp) [MPa·µm^(1/2)]', fontsize=12)
ax.set_title(f'k_HP(comp) Discovery: R²={r2_kHP:.3f}', fontsize=13)
ax.grid(True, alpha=0.3)

# (d) Combined model parity
ax = axes[1, 1]
ax.scatter(y, ys_combined, c='purple', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
lims = [min(y.min(), ys_combined.min()) * 0.9, max(y.max(), ys_combined.max()) * 1.1]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Experimental YS (MPa)', fontsize=12)
ax.set_ylabel('σ₀(comp) + k_HP(comp)·d⁻¹/² [MPa]', fontsize=12)
ax.set_title(f'Combined: R²={r2_combined:.3f}, RMSE={rmse_combined:.1f}', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle('PySR Symbolic Regression: Interpretable Strengthening Laws', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/15_pysr_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved PySR results plot")

# ============================================================
# EQUATION COMPLEXITY-ACCURACY TRADEOFF
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
eqs = model_full.equations_
ax.plot(eqs['complexity'], eqs['loss'], 'o-', color='steelblue', markersize=8)
ax.set_xlabel('Equation Complexity', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title('PySR Pareto Front: Complexity vs Accuracy', fontsize=14)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Annotate a few key equations
for _, eq in eqs.iterrows():
    if eq['complexity'] in [3, 7, 12, 20]:
        ax.annotate(eq['equation'][:40], (eq['complexity'], eq['loss']),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=7, alpha=0.8)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/16_pysr_pareto.png', dpi=150)
plt.close()
print("  Saved Pareto front plot")

print("\n" + "=" * 70)
print("PySR ANALYSIS COMPLETE")
print("=" * 70)
