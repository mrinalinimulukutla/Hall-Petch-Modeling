#!/usr/bin/env python3
"""
VIF + Condition-Number Diagnostics
===================================
Reports Variance Inflation Factor and design-matrix condition number for
the feature sets used by:

  - PCA-OLS pre-curation (full Wen library)
  - PCA-OLS post-curation (curated subset)
  - PCA-OLS post-PCA (principal components)
  - M3 (8 elements + d^-1/2)
  - INTERACTIONS_ALT (64-feature pool used by XGBoost)
  - F3 (the curated Wen + grain + processing set fed to PySR)

The pre/post comparison is the visual argument that PCA "fixes" the
multicollinearity problem.

Outputs
-------
  results/vif_diagnostics.csv
  analysis_plots/85_vif_diagnostics.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv').dropna(subset=['YS']).reset_index(drop=True)
SD_COL = next((c for c in ('SD_GS', 'GrainSize_SD', 'SD_grain', 'GS_SD') if c in df.columns), None)

FEATURE_SETS = {
    'M3 (7 elements + d^-1/2)':
        [f'{e}_frac' for e in ELEMENTS if e != 'Ni'] + ['d_inv_sqrt'],
    'F3 (Wen + grain + proc)':
        ['VEC', 'dH_mix', 'delta_chi', 'Omega', 'dS_mix', 'delta',
         'd_inv_sqrt', SD_COL, 'ColdWork', 'RecrystT', 'HoldTime']
        if SD_COL else
        ['VEC', 'dH_mix', 'delta_chi', 'Omega', 'dS_mix', 'delta',
         'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Curated Wen + SD + 1/sqrt(d)':
        ['VEC', 'dH_mix', 'dS_mix', 'Omega', 'delta_chi', 'delta',
         'd_inv_sqrt'] + ([SD_COL] if SD_COL else []),
}

def vif_for(cols):
    cols = [c for c in cols if c in df.columns]
    X = StandardScaler().fit_transform(df[cols].values.astype(float))
    vif = pd.DataFrame({
        'feature': cols,
        'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])],
    })
    cond = float(np.linalg.cond(X))
    return vif.sort_values('VIF', ascending=False), cond

rows = []
for name, cols in FEATURE_SETS.items():
    print(f"\n--- {name} ---")
    vif, cond = vif_for(cols)
    print(vif.to_string(index=False))
    print(f"condition number = {cond:.2f}")
    rows.append({
        'feature_set': name,
        'n_features': len(vif),
        'max_VIF': float(vif['VIF'].max()),
        'mean_VIF': float(vif['VIF'].mean()),
        'condition_number': cond,
        'n_above_10': int((vif['VIF'] > 10).sum()),
    })

# Post-PCA on curated set
curated = [c for c in FEATURE_SETS['Curated Wen + SD + 1/sqrt(d)']
           if c in df.columns]
X_std = StandardScaler().fit_transform(df[curated].values.astype(float))
pca = PCA(n_components=min(6, len(curated))).fit_transform(X_std)
pca_vif = pd.DataFrame({
    'feature': [f'PC{i+1}' for i in range(pca.shape[1])],
    'VIF': [variance_inflation_factor(pca, i) for i in range(pca.shape[1])],
})
print("\n--- Curated set after PCA ---")
print(pca_vif.to_string(index=False))
rows.append({
    'feature_set': 'Curated Wen + SD (after PCA)',
    'n_features': pca.shape[1],
    'max_VIF': float(pca_vif['VIF'].max()),
    'mean_VIF': float(pca_vif['VIF'].mean()),
    'condition_number': float(np.linalg.cond(pca)),
    'n_above_10': int((pca_vif['VIF'] > 10).sum()),
})

pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/vif_diagnostics.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/vif_diagnostics.csv")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar([r['feature_set'] for r in rows],
       [r['max_VIF'] for r in rows], color='#4c72b0', edgecolor='black')
ax.axhline(10, color='red', linestyle='--', label='VIF = 10 threshold')
ax.set_xticklabels([r['feature_set'] for r in rows], rotation=20, ha='right')
ax.set_ylabel('max VIF')
ax.set_title('Multicollinearity diagnostics across feature sets')
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/85_vif_diagnostics.png', dpi=150)
plt.close()
print(f"Wrote {PLOTS_DIR}/85_vif_diagnostics.png")
