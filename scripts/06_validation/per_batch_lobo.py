#!/usr/bin/env python3
"""
Per-Batch LOBO Breakdown
=========================
The baseline analysis reports aggregate LOBO R². But the six batches (BBA, BBB, BBC,
CBA, CBB, CBC) sample qualitatively different regions of composition space
(see scripts/eda_diagnostics.py and the pre-modeling diagnostics). Aggregate LOBO averages
over those qualitatively different tests; per-batch LOBO is more honest.

For each candidate model and each held-out batch, report:

  - n_train, n_test
  - R²(test), RMSE(test)
  - extrapolation distance: convex-hull containment of held-out points
    in PC1–PC2 space of the training composition

Outputs
-------
  results/per_batch_lobo.csv
  analysis_plots/86_per_batch_lobo.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial import ConvexHull, Delaunay

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv').dropna(subset=['YS']).reset_index(drop=True)
if 'Iteration' not in df.columns:
    raise SystemExit("[fatal] no 'Iteration' column — per-batch LOBO impossible.")
batches = sorted(df['Iteration'].unique())
print(f"Batches found: {batches}")

# Two simple baseline models; the full M3 / PCA-OLS / SR equations should
# be added when their refit code exists in this repo.
MODELS = {
    'classical_HP': (['d_inv_sqrt'], LinearRegression),
    'M3_lite (7 elem + d^-1/2)':
        ([f'{e}_frac' for e in ELEMENTS if e != 'Ni'] + ['d_inv_sqrt'],
         LinearRegression),
}

def hull_containment(train_pca, test_pca):
    """Fraction of test points inside the convex hull of training points."""
    try:
        hull = Delaunay(train_pca[:, :2])
        inside = hull.find_simplex(test_pca[:, :2]) >= 0
        return float(inside.mean())
    except Exception:
        return np.nan

# PCA in composition space for the extrapolation diagnostic
comp = df[[f'{e}_frac' for e in ELEMENTS]].values
comp_pca_full = PCA(n_components=min(6, comp.shape[1])).fit_transform(
    StandardScaler().fit_transform(comp))

rows = []
for model_name, (cols, ctor) in MODELS.items():
    print(f"\n--- {model_name} ---")
    X = df[cols].values.astype(float)
    y = df['YS'].values.astype(float)
    for held in batches:
        te = df['Iteration'].values == held
        tr = ~te
        m = ctor().fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r2 = r2_score(y[te], pred) if te.sum() > 1 else np.nan
        rmse = float(np.sqrt(mean_squared_error(y[te], pred))) if te.sum() else np.nan
        hull_frac = hull_containment(comp_pca_full[tr], comp_pca_full[te])
        rows.append({
            'model': model_name, 'held_out_batch': held,
            'n_train': int(tr.sum()), 'n_test': int(te.sum()),
            'R2_test': r2, 'RMSE_test': rmse,
            'hull_containment_frac': hull_frac,
        })
        print(f"  hold-out {held}: n_test = {te.sum():2d}, "
              f"R² = {r2:6.3f}, RMSE = {rmse:6.1f}, "
              f"hull-in = {hull_frac:.2f}")

out = pd.DataFrame(rows)
out.to_csv(f'{RESULTS_DIR}/per_batch_lobo.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/per_batch_lobo.csv")

fig, ax = plt.subplots(figsize=(9, 4))
for model_name in MODELS:
    sub = out[out['model'] == model_name]
    ax.bar(sub['held_out_batch'] + ' / ' + model_name[:6],
           sub['R2_test'], label=model_name, alpha=0.8, edgecolor='black')
ax.set_ylabel('R²(held-out batch)')
ax.set_title('Per-batch LOBO breakdown')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/86_per_batch_lobo.png', dpi=150)
plt.close()
print(f"Wrote {PLOTS_DIR}/86_per_batch_lobo.png")
