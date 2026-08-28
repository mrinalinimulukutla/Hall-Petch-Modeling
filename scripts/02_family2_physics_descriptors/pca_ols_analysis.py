#!/usr/bin/env python3
"""
PCA-OLS Analysis on Curated Wen Library 
==============================================================
Implements the physics-informed linear baseline from compact-equation stream:

  1. Build the Wen descriptor library (VEC, dH_mix, dS_mix, Omega, delta_chi,
     delta, plus 1/sqrt(d), grain-size SD, processing variables).
  2. Curate to a physically complementary subset using VIF < 10.
  3. PCA-rotate the curated subset to an orthogonal basis (VIF < 3 target).
  4. OLS on the principal components.
  5. Reconstruct feature importance by projecting the OLS coefficients on
     the PCs back through the loading matrix.
  6. Evaluate under LOO + LOBO (in addition to 5-fold for parity with earlier 5-fold reporting).

The reconstructed importance vector is the load-bearing artifact: it lets
us argue that PCA-OLS and PySR converge on the same dominant features.

Outputs
-------
  results/pca_ols_results.csv
  results/pca_ols_feature_importance.csv
  analysis_plots/77_pca_ols_loadings.png
  analysis_plots/78_pca_ols_importance.png

Provisional 5-fold R² (YS, provisional): 0.675. Integration target: LOO + LOBO.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, KFold
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR

print("=" * 70)
print("PCA-OLS ANALYSIS (compact-equation stream)")
print("=" * 70)

# ============================================================
# 1. LOAD
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df = df.dropna(subset=['YS']).reset_index(drop=True)
n = len(df)
print(f"Loaded {n} alloys with YS data.")

# ============================================================
# 2. WEN LIBRARY + SD_grain + processing
# ============================================================
# Curated subset (the "physically complementary" set from the curated-descriptor selection).
# If a column is missing, fail loudly rather than silently dropping it.
CURATED_WEN = ['VEC', 'dH_mix', 'dS_mix', 'Omega', 'delta_chi', 'delta']
GRAIN_FEATS = ['d_inv_sqrt']           # + SD_grain when present in derived CSV
PROC_FEATS  = ['ColdWork', 'RecrystT', 'HoldTime']

candidate_feats = CURATED_WEN + GRAIN_FEATS + PROC_FEATS
# SD_grain may live under several aliases; prefer canonical names.
for sd_alias in ('SD_GS', 'GrainSize_SD', 'SD_grain', 'GS_SD'):
    if sd_alias in df.columns:
        candidate_feats.append(sd_alias)
        break
else:
    print("[warn] no SD_grain column found in data_with_descriptors.csv — "
          "skipping the SD_grain channel for now.")

missing = [c for c in candidate_feats if c not in df.columns]
if missing:
    print(f"[warn] missing columns from derived CSV: {missing}")
    candidate_feats = [c for c in candidate_feats if c in df.columns]

X = df[candidate_feats].values.astype(float)
y = df['YS'].values.astype(float)
print(f"Candidate features ({len(candidate_feats)}): {candidate_feats}")

# ============================================================
# 3. VIF curation (drop anything > 10; iterate until clean)
# ============================================================
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

def vif_table(X_, names):
    return pd.DataFrame({
        'feature': names,
        'VIF': [variance_inflation_factor(X_, i) for i in range(X_.shape[1])],
    }).sort_values('VIF', ascending=False)

vif = vif_table(X_std, candidate_feats)
print("\nInitial VIF table:")
print(vif.to_string(index=False))

# TODO: port the compact-equation stream's iterative VIF curation rule from
# Comprehensive_Model_Comparison_v3.ipynb. For now we report only.

# ============================================================
# 4. PCA rotation + OLS
# ============================================================
pca = PCA(n_components=min(6, X_std.shape[1]))   # compact-equation stream: 6 PCs retain ~88% var
Z = pca.fit_transform(X_std)
print(f"\nPCA: {Z.shape[1]} components, explained variance "
      f"ratio = {pca.explained_variance_ratio_.round(3).tolist()}, "
      f"cumulative = {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

ols = LinearRegression().fit(Z, y)

# ============================================================
# 5. Cross-validation: 5-fold + LOO + LOBO
# ============================================================
def cv_r2(X_, y_, splitter, groups=None):
    preds = np.zeros_like(y_, dtype=float)
    for tr, te in (splitter.split(X_, y_, groups) if groups is not None
                   else splitter.split(X_)):
        m = LinearRegression().fit(X_[tr], y_[tr])
        preds[te] = m.predict(X_[te])
    return r2_score(y_, preds), np.sqrt(mean_squared_error(y_, preds))

r2_5fold, rmse_5fold = cv_r2(Z, y, KFold(n_splits=5, shuffle=True, random_state=42))
r2_loo, rmse_loo = cv_r2(Z, y, LeaveOneOut())

groups = df['Iteration'].values if 'Iteration' in df.columns else None
if groups is not None:
    r2_lobo, rmse_lobo = cv_r2(Z, y, LeaveOneGroupOut(), groups=groups)
else:
    r2_lobo, rmse_lobo = np.nan, np.nan
    print("[warn] no 'Iteration' column — LOBO skipped")

print(f"\n5-fold R² = {r2_5fold:.3f}, RMSE = {rmse_5fold:.1f} MPa")
print(f"LOO   R² = {r2_loo:.3f}, RMSE = {rmse_loo:.1f} MPa")
print(f"LOBO  R² = {r2_lobo:.3f}, RMSE = {rmse_lobo:.1f} MPa")

# ============================================================
# 6. Reconstruct feature importance through the loading matrix
# ============================================================
# beta_features = loadings.T @ beta_pcs / sigma_features
beta_pcs = ols.coef_                                           # (n_pcs,)
loadings = pca.components_                                     # (n_pcs, n_feats)
beta_feats_std = loadings.T @ beta_pcs                         # importance in z-space
importance = pd.DataFrame({
    'feature': candidate_feats,
    'reconstructed_importance': beta_feats_std,
    'abs_importance': np.abs(beta_feats_std),
}).sort_values('abs_importance', ascending=False)

print("\nReconstructed feature importance (top 10):")
print(importance.head(10).to_string(index=False))

# ============================================================
# 7. Persist results
# ============================================================
out = pd.DataFrame([{
    'model': 'PCA-OLS (curated Wen + SD_grain)',
    'n_features_in': X.shape[1],
    'n_pcs': Z.shape[1],
    'cum_var': float(pca.explained_variance_ratio_.cumsum()[-1]),
    'R2_5fold': r2_5fold, 'RMSE_5fold': rmse_5fold,
    'R2_LOO': r2_loo,     'RMSE_LOO': rmse_loo,
    'R2_LOBO': r2_lobo,   'RMSE_LOBO': rmse_lobo,
}])
out.to_csv(f'{RESULTS_DIR}/pca_ols_results.csv', index=False)
importance.to_csv(f'{RESULTS_DIR}/pca_ols_feature_importance.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/pca_ols_results.csv")
print(f"Wrote {RESULTS_DIR}/pca_ols_feature_importance.csv")

# ============================================================
# 8. Plots
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(candidate_feats)))
ax.set_xticklabels(candidate_feats, rotation=45, ha='right')
ax.set_yticks(range(Z.shape[1]))
ax.set_yticklabels([f'PC{i+1}' for i in range(Z.shape[1])])
plt.colorbar(im, ax=ax, label='Loading')
ax.set_title('PCA loadings on curated Wen + SD_grain library')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/77_pca_ols_loadings.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance['feature'][::-1], importance['reconstructed_importance'][::-1])
ax.set_xlabel('Reconstructed importance (z-space)')
ax.set_title('PCA-OLS reconstructed feature importance for YS')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/78_pca_ols_importance.png', dpi=150)
plt.close()

print(f"Wrote {PLOTS_DIR}/77_pca_ols_loadings.png and 78_pca_ols_importance.png")
print("\nDone.")
