#!/usr/bin/env python3
"""
Combined PySR + SISSO Pareto figure for §4.8 of the paper.

Fits the PySR "full" model once (same hyperparameters as pysr_analysis.py
strategy 1), then overlays SISSO v1 (Eq.~\\ref{eq:sisso}) and SISSO v2
BIC-landscape points on the same (complexity, MSE) axes.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
import _figstyle as S
S.apply()
BASE = str(REPO_ROOT)
OUT_FIG = f'{PAPER_FIG_DIR}/fig08_symbolic_pareto.png'
OUT_CSV = f'{RESULTS_DIR}/pysr_pareto_full.csv'

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

print("=" * 70)
print("PySR + SISSO PARETO FIGURE")
print("=" * 70)

# --- Load data
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df_ys = df.dropna(subset=['YS'])
feature_cols = [f'{el}_frac' for el in ELEMENTS] + ['d_inv_sqrt', 'ColdWork', 'RecrystT']
X = df_ys[feature_cols].values
y = df_ys['YS'].values
print(f"Loaded {len(y)} alloys with YS data; {X.shape[1]} features")

# --- Fit PySR (full model only)
from pysr import PySRRegressor

model_full = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    populations=20,
    population_size=40,
    maxsize=25,
    maxdepth=6,
    parsimony=0.005,
    nested_constraints={"sqrt": {"sqrt": 0}, "square": {"square": 0}},
    model_selection="best",
    temp_equation_file=True,
    tempdir=f'{BASE}/pysr_temp_combined',
    verbosity=1,
    progress=False,
    random_state=42,
    deterministic=True,
    parallelism='serial',
    turbo=False,
    batching=False,
)
print("\nFitting PySR full model...")
model_full.fit(X, y, variable_names=feature_cols)
eqs = model_full.equations_.copy()
print(f"  PySR Pareto: {len(eqs)} equations, complexity {eqs['complexity'].min()}--{eqs['complexity'].max()}")
eqs[['complexity', 'loss', 'score', 'equation']].to_csv(OUT_CSV, index=False)
print(f"  Saved {OUT_CSV}")

# --- SISSO data
# v1 Full (Eq. eq:sisso): k_eff=4, Train RMSE=46.95 (from sisso_results.csv)
sisso_v1 = pd.read_csv(f'{RESULTS_DIR}/sisso_results.csv')
sisso_full = sisso_v1[sisso_v1['Model'] == 'SISSO Full'].iloc[0]
sisso_v1_pt = (int(sisso_full['k_eff']), float(sisso_full['LOO_RMSE'])**2)
# Convert from LOO_RMSE to Train MSE for fair comparison with PySR's training MSE
# Train_R2 column is also there:
train_r2 = float(sisso_full['Train_R2'])
y_var = np.var(y)
sisso_v1_train_mse = (1.0 - train_r2) * y_var
sisso_v1_pt = (int(sisso_full['k_eff']), sisso_v1_train_mse)
print(f"\nSISSO Full v1: k_eff={sisso_v1_pt[0]}, train MSE={sisso_v1_pt[1]:.0f}")

# v2 BIC landscape: dim 1..4 -> k_eff 2..5
v2 = pd.read_csv(f'{RESULTS_DIR}/sisso_v2_bic_landscape.csv')
v2['train_MSE'] = v2['Train_RMSE'] ** 2
sisso_v2_pts = list(zip(v2['k_eff'].astype(int), v2['train_MSE']))
print(f"SISSO v2 landscape: {sisso_v2_pts}")

# --- Build figure
fig, ax = plt.subplots(figsize=(5.0, 4.6))

# PySR Pareto front
ax.plot(eqs['complexity'], eqs['loss'], 'o-', color=S.LINEAR,
        markersize=7, linewidth=1.6, label='PySR Pareto front', zorder=2)

# SISSO v1 Full point
ax.scatter([sisso_v1_pt[0]], [sisso_v1_pt[1]],
           marker='*', s=240, color=S.HIGHLIGHT, edgecolor='black',
           linewidth=1.0, label='SISSO 3-term (Eq. 4)', zorder=4)

# SISSO v2 landscape points
xs = [p[0] for p in sisso_v2_pts]
ys = [p[1] for p in sisso_v2_pts]
ax.scatter(xs, ys, marker='s', s=80, color=S.OKABE['green'], edgecolor='black',
           linewidth=0.8, label='SISSO v2 (dim 1--4)', zorder=3)

ax.set_xlabel('Equation complexity / $k_\\mathrm{eff}$', fontsize=14)
ax.set_ylabel('Training loss (MSE, MPa$^2$)', fontsize=14)
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(left=0)
plt.savefig(OUT_FIG, dpi=200, bbox_inches='tight')
print(f"\nSaved {OUT_FIG}")
print("Done.")
