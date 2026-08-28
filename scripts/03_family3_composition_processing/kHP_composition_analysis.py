#!/usr/bin/env python3
"""
Refined Analysis of Composition Dependence of k_HP
====================================================
Two-stage approach:
  Stage 1: Use M3 (σ₀(all elem) + constant k) to estimate σ₀(comp) per alloy
  Stage 2: Compute effective k_HP per alloy, then regress on composition

Also fits a full Bayesian model: σ₀(all) + k(comp)·d⁻¹/²
to quantify uncertainty in the composition dependence of k.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az
import pymc as pm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("REFINED ANALYSIS: COMPOSITION DEPENDENCE OF k_HP")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df = df.dropna(subset=['YS'])
n = len(df)

y = df['YS'].values.astype(float)
d = df['GrainSize'].values.astype(float)
d_inv_sqrt = d ** -0.5

elem_names = ['Al_frac', 'Co_frac', 'Cr_frac', 'Cu_frac', 'Fe_frac', 'Mn_frac', 'V_frac']
elem_short = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'V']
X_elem = df[elem_names].values.astype(float)

V = df['V_frac'].values
Mn = df['Mn_frac'].values
Al = df['Al_frac'].values
Ni = df['Ni_frac'].values

# ============================================================
# 2. STAGE 1: Fit M3 (σ₀(all elem) + constant k)
# ============================================================
print("\n" + "=" * 70)
print("STAGE 1: σ₀(comp) from M3 (all elements + constant k)")
print("=" * 70)

ones = np.ones(n)
X_m3 = np.column_stack([ones, X_elem, d_inv_sqrt])
beta_m3 = np.linalg.lstsq(X_m3, y, rcond=None)[0]

sigma0_labels = ['σ₀₀'] + [f'α_{e}' for e in elem_short]
print("\n  σ₀(comp) = σ₀₀ + Σ αᵢ·xᵢ")
for lbl, b in zip(sigma0_labels, beta_m3[:8]):
    print(f"    {lbl:>6s} = {b:>8.1f}")
print(f"    {'k':>6s} = {beta_m3[-1]:>8.1f} (global constant)")

sigma0_comp = X_m3[:, :8] @ beta_m3[:8]
k_global = beta_m3[-1]

# ============================================================
# 3. STAGE 2: Effective k_HP per alloy
# ============================================================
print("\n" + "=" * 70)
print("STAGE 2: Effective k_HP per alloy")
print("=" * 70)

# k_eff_i = (YS_i - σ₀(comp_i)) / d_i^{-1/2}
k_eff = (y - sigma0_comp) / d_inv_sqrt

print(f"\n  k_eff range: {k_eff.min():.0f} – {k_eff.max():.0f} MPa·µm¹/²")
print(f"  k_eff mean:  {k_eff.mean():.0f} ± {k_eff.std():.0f} MPa·µm¹/²")
print(f"  k_global:    {k_global:.0f} MPa·µm¹/²")
print(f"  CV:          {k_eff.std() / k_eff.mean() * 100:.1f}%")

# ============================================================
# 4. CORRELATIONS: k_eff vs composition
# ============================================================
print("\n" + "=" * 70)
print("CORRELATIONS: k_eff vs COMPOSITION")
print("=" * 70)

print(f"\n  {'Element':>8s} {'Pearson r':>10s} {'p-value':>10s} {'Signif':>8s}")
print("  " + "-" * 42)
corr_results = []
for i, elem in enumerate(elem_short):
    r, p = stats.pearsonr(X_elem[:, i], k_eff)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {elem:>8s} {r:>10.3f} {p:>10.4f} {sig:>8s}")
    corr_results.append((elem, r, p))

# Also check Ni
r_ni, p_ni = stats.pearsonr(Ni, k_eff)
sig_ni = '***' if p_ni < 0.001 else '**' if p_ni < 0.01 else '*' if p_ni < 0.05 else 'ns'
print(f"  {'Ni':>8s} {r_ni:>10.3f} {p_ni:>10.4f} {sig_ni:>8s}")

# Check physics descriptors
for desc_name, desc_label in [('delta', 'δ'), ('Omega', 'Ω'), ('VEC', 'VEC'),
                                ('Phi_VLC', 'Φ_VLC'), ('dS_mix', 'ΔS_mix')]:
    if desc_name in df.columns:
        desc_vals = df[desc_name].values
        r, p = stats.pearsonr(desc_vals, k_eff)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {desc_label:>8s} {r:>10.3f} {p:>10.4f} {sig:>8s}")

# ============================================================
# 5. REGRESSION: k_eff = k₀ + Σ βᵢ·xᵢ
# ============================================================
print("\n" + "=" * 70)
print("REGRESSION: k_eff = k₀ + Σ βᵢ·xᵢ")
print("=" * 70)

# Full regression on all 7 elements
X_k = np.column_stack([ones, X_elem])
beta_k = np.linalg.lstsq(X_k, k_eff, rcond=None)[0]
k_pred = X_k @ beta_k
ss_res = np.sum((k_eff - k_pred) ** 2)
ss_tot = np.sum((k_eff - k_eff.mean()) ** 2)
r2_k = 1 - ss_res / ss_tot

# Standard errors via (X'X)^{-1} · σ²
sigma2_resid = ss_res / (n - X_k.shape[1])
cov_beta = sigma2_resid * np.linalg.inv(X_k.T @ X_k)
se_beta = np.sqrt(np.diag(cov_beta))
t_stats = beta_k / se_beta
p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n - X_k.shape[1])

# F-test for overall significance
f_stat = (r2_k / 7) / ((1 - r2_k) / (n - 8))
p_f = stats.f.sf(f_stat, 7, n - 8)

print(f"\n  k_eff = k₀ + Σ βᵢ·xᵢ  (OLS, n={n})")
print(f"  R² = {r2_k:.3f},  F({7},{n-8}) = {f_stat:.2f},  p = {p_f:.4f}")
print(f"\n  {'Param':>8s} {'Coeff':>10s} {'SE':>8s} {'t':>7s} {'p':>8s} {'Signif':>6s}")
print("  " + "-" * 52)
labels_k = ['k₀'] + [f'β_{e}' for e in elem_short]
for lbl, b, se, t, p in zip(labels_k, beta_k, se_beta, t_stats, p_vals):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {lbl:>8s} {b:>10.1f} {se:>8.1f} {t:>7.2f} {p:>8.4f} {sig:>6s}")

# Stepwise: just significant elements
print("\n  Interpretation:")
for lbl, b, p in zip(labels_k[1:], beta_k[1:], p_vals[1:]):
    if p < 0.1:
        elem = lbl.split('_')[1]
        direction = "increases" if b > 0 else "decreases"
        print(f"    {elem}: {direction} k_HP by {abs(b):.0f} MPa·µm¹/² per unit fraction (p={p:.3f})")

# ============================================================
# 6. SPARSE MODELS: k_eff on subsets of elements
# ============================================================
print("\n" + "=" * 70)
print("SPARSE k_HP MODELS")
print("=" * 70)

sparse_models = {
    'k = const': np.column_stack([ones]),
    'k(V)': np.column_stack([ones, V]),
    'k(Mn)': np.column_stack([ones, Mn]),
    'k(Al)': np.column_stack([ones, Al]),
    'k(V,Mn)': np.column_stack([ones, V, Mn]),
    'k(V,Al)': np.column_stack([ones, V, Al]),
    'k(V,Mn,Al)': np.column_stack([ones, V, Mn, Al]),
    'k(all 7)': np.column_stack([ones, X_elem]),
}

print(f"\n  {'Model':<20s} {'R²':>6s} {'Adj R²':>7s} {'RMSE':>8s} {'BIC':>8s}")
print("  " + "-" * 55)
for name, X in sparse_models.items():
    b = np.linalg.lstsq(X, k_eff, rcond=None)[0]
    pred = X @ b
    ss_r = np.sum((k_eff - pred) ** 2)
    ss_t = np.sum((k_eff - k_eff.mean()) ** 2)
    r2 = 1 - ss_r / ss_t
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(ss_r / n)
    bic = n * np.log(ss_r / n) + p * np.log(n)
    print(f"  {name:<20s} {r2:>6.3f} {adj_r2:>7.3f} {rmse:>8.0f} {bic:>8.1f}")

# ============================================================
# 7. BAYESIAN: Full model σ₀(all) + k(comp)·d⁻¹/²
# ============================================================
print("\n" + "=" * 70)
print("BAYESIAN: σ₀(all elem) + k(V,Mn)·d⁻¹/²")
print("=" * 70)

# Model: σ_y = [σ₀₀ + Σαᵢ·xᵢ] + [k₀ + β_V·V + β_Mn·Mn]·d⁻¹/²
X_bayes = np.column_stack([ones, X_elem, d_inv_sqrt, V * d_inv_sqrt, Mn * d_inv_sqrt])
n_feat = X_bayes.shape[1]
bayes_labels = ['σ₀₀'] + [f'α_{e}' for e in elem_short] + ['k₀', 'β_V', 'β_Mn']

print(f"\n  Fitting: σ₀(7 elem) + k(V,Mn)·d⁻¹/² ({n_feat} coefficients + σ) ...")

with pm.Model() as model_refined:
    beta = pm.Normal('beta', mu=0, sigma=1000, shape=n_feat)
    sigma = pm.HalfCauchy('sigma', beta=50)
    mu = pm.math.dot(X_bayes, beta)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    trace_refined = pm.sample(4000, tune=2000, cores=2, chains=2,
                              random_seed=42, progressbar=True,
                              return_inferencedata=True)
    pm.compute_log_likelihood(trace_refined, model=model_refined)

rhat_max = float(az.rhat(trace_refined)['beta'].max())
ess_min = float(az.ess(trace_refined)['beta'].min())
print(f"    R-hat max: {rhat_max:.4f}, ESS min: {ess_min:.0f}")

# Extract posteriors
beta_post = trace_refined.posterior['beta'].values.reshape(-1, n_feat)
beta_mean = beta_post.mean(axis=0)
beta_std = beta_post.std(axis=0)

print(f"\n  {'Param':>8s} {'Mean':>10s} {'Std':>8s} {'94% HDI':>24s} {'P(>0)':>7s}")
print("  " + "-" * 62)
hdi_vals = az.hdi(trace_refined, var_names=['beta'])['beta'].values
for i, lbl in enumerate(bayes_labels):
    p_pos = (beta_post[:, i] > 0).mean()
    print(f"  {lbl:>8s} {beta_mean[i]:>10.1f} {beta_std[i]:>8.1f} "
          f"[{hdi_vals[i, 0]:>8.1f}, {hdi_vals[i, 1]:>8.1f}] {p_pos:>7.3f}")

sigma_mean = trace_refined.posterior['sigma'].values.mean()
print(f"  {'σ':>8s} {sigma_mean:>10.1f}")

# ============================================================
# 8. COMPUTE PER-ALLOY k_HP FROM BAYESIAN MODEL
# ============================================================
print("\n" + "=" * 70)
print("PER-ALLOY k_HP FROM BAYESIAN MODEL")
print("=" * 70)

# k_HP(comp) = k₀ + β_V·V + β_Mn·Mn  (posterior samples)
k0_idx = bayes_labels.index('k₀')
bV_idx = bayes_labels.index('β_V')
bMn_idx = bayes_labels.index('β_Mn')

# Per-alloy k_HP: posterior distribution for each alloy
k_hp_samples = (beta_post[:, k0_idx:k0_idx+1] +
                beta_post[:, bV_idx:bV_idx+1] * V[np.newaxis, :] +
                beta_post[:, bMn_idx:bMn_idx+1] * Mn[np.newaxis, :])

k_hp_mean = k_hp_samples.mean(axis=0)
k_hp_lo = np.percentile(k_hp_samples, 3, axis=0)
k_hp_hi = np.percentile(k_hp_samples, 97, axis=0)

print(f"\n  k_HP posterior mean range: {k_hp_mean.min():.0f} – {k_hp_mean.max():.0f} MPa·µm¹/²")
print(f"  k_HP grand mean:          {k_hp_mean.mean():.0f} ± {k_hp_mean.std():.0f} MPa·µm¹/²")

# Compare with literature
print("\n  Per-alloy k_HP vs literature benchmarks:")
print(f"    Pure Cu:        110")
print(f"    Pure Ni:        160")
print(f"    316L SS:        322")
print(f"    CoCrFeMnNi:     494  (Otto et al., 2013)")
print(f"    CoCrNi:         677  (Yoshida et al., 2017)")
print(f"    Our range:      {k_hp_mean.min():.0f} – {k_hp_mean.max():.0f}")
print(f"    Our mean:       {k_hp_mean.mean():.0f}")

# ============================================================
# 9. COMPARE MODELS: constant k vs k(V,Mn)
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON: constant k vs k(V,Mn)")
print("=" * 70)

# Also fit M3 in PyMC for fair comparison
X_m3_full = np.column_stack([ones, X_elem, d_inv_sqrt])
with pm.Model() as model_m3:
    beta_m3b = pm.Normal('beta', mu=0, sigma=1000, shape=X_m3_full.shape[1])
    sigma_m3b = pm.HalfCauchy('sigma', beta=50)
    mu_m3b = pm.math.dot(X_m3_full, beta_m3b)
    y_obs_m3b = pm.Normal('y_obs', mu=mu_m3b, sigma=sigma_m3b, observed=y)
    trace_m3 = pm.sample(4000, tune=2000, cores=2, chains=2,
                         random_seed=42, progressbar=True,
                         return_inferencedata=True)
    pm.compute_log_likelihood(trace_m3, model=model_m3)

comparison = az.compare(
    {'M3: σ₀(all)+k=const': trace_m3,
     'M3+: σ₀(all)+k(V,Mn)': trace_refined},
    ic='loo', method='stacking', scale='log'
)
print("\n" + str(comparison))

# ============================================================
# 10. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# --- Plot 41: k_eff vs composition (scatter grid) ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, (elem, ax) in enumerate(zip(elem_short, axes[:7])):
    x_vals = X_elem[:, i] * 100
    ax.scatter(x_vals, k_eff, c=d, cmap='viridis', s=30, edgecolors='k', linewidth=0.3, alpha=0.8)
    ax.axhline(y=k_global, color='red', linestyle='--', linewidth=1, label=f'k_global={k_global:.0f}')

    # Linear fit
    if x_vals.std() > 0:
        slope, intercept, r, p, se = stats.linregress(x_vals, k_eff)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, 'b-', linewidth=1.5, alpha=0.7)
        sig = '*' if p < 0.05 else ''
        ax.set_title(f'{elem} (r={r:.2f}, p={p:.3f}){sig}', fontsize=10)
    else:
        ax.set_title(f'{elem}', fontsize=10)

    ax.set_xlabel(f'{elem} (at%)')
    ax.set_ylabel('k_eff (MPa·µm¹/²)')
    ax.set_ylim(k_eff.min() - 200, k_eff.max() + 200)

# Literature reference lines in last subplot
ax_lit = axes[7]
lit_data = [
    ('Cu', 110), ('Ni', 160), ('316L', 322), ('CoCrFeMnNi', 494),
    ('CoCrNi', 677), ('This work\n(mean)', k_eff.mean()),
]
colors_lit = ['#E6E6E6', '#C0C0C0', '#90CAF9', '#4CAF50', '#FF9800', '#F44336']
bars = ax_lit.barh([x[0] for x in lit_data], [x[1] for x in lit_data],
                    color=colors_lit, edgecolor='k', linewidth=0.5)
ax_lit.set_xlabel('k_HP (MPa·µm¹/²)')
ax_lit.set_title('Literature comparison')
for bar, (name, val) in zip(bars, lit_data):
    ax_lit.text(val + 10, bar.get_y() + bar.get_height()/2, f'{val:.0f}',
                va='center', fontsize=9)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=d.min(), vmax=d.max()))
sm.set_array([])
fig.colorbar(sm, ax=axes[6], label='Grain size (µm)', shrink=0.7)

plt.suptitle('Effective k_HP vs Element Content (colored by grain size)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/41_kHP_vs_composition.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 41_kHP_vs_composition.png")

# --- Plot 42: Bayesian k_HP posterior per alloy ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 42a: k_HP(comp) vs V
ax = axes[0]
sort_idx = np.argsort(V)
ax.fill_between(V[sort_idx] * 100, k_hp_lo[sort_idx], k_hp_hi[sort_idx],
                alpha=0.3, color='steelblue', label='94% CI')
ax.scatter(V * 100, k_hp_mean, c=Mn * 100, cmap='coolwarm', s=40,
           edgecolors='k', linewidth=0.3, zorder=5)
ax.axhline(y=494, color='green', linestyle=':', linewidth=1.5, label='CoCrFeMnNi (494)')
ax.axhline(y=677, color='orange', linestyle=':', linewidth=1.5, label='CoCrNi (677)')
ax.set_xlabel('V content (at%)')
ax.set_ylabel('k_HP (MPa·µm¹/²)')
ax.set_title('(a) k_HP vs V (colored by Mn)')
ax.legend(fontsize=8, loc='upper left')
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=Mn.min()*100, vmax=Mn.max()*100))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Mn (at%)', shrink=0.8)

# 42b: k_HP(comp) vs Mn
ax = axes[1]
ax.scatter(Mn * 100, k_hp_mean, c=V * 100, cmap='plasma', s=40,
           edgecolors='k', linewidth=0.3, zorder=5)
ax.axhline(y=494, color='green', linestyle=':', linewidth=1.5, label='CoCrFeMnNi (494)')
ax.axhline(y=677, color='orange', linestyle=':', linewidth=1.5, label='CoCrNi (677)')
ax.set_xlabel('Mn content (at%)')
ax.set_ylabel('k_HP (MPa·µm¹/²)')
ax.set_title('(b) k_HP vs Mn (colored by V)')
ax.legend(fontsize=8, loc='upper left')
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=V.min()*100, vmax=V.max()*100))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='V (at%)', shrink=0.8)

# 42c: Posterior of β_V and β_Mn
ax = axes[2]
ax.hist(beta_post[:, bV_idx], bins=60, alpha=0.6, color='steelblue',
        density=True, label=f'β_V = {beta_mean[bV_idx]:.0f} ± {beta_std[bV_idx]:.0f}')
ax.hist(beta_post[:, bMn_idx], bins=60, alpha=0.6, color='coral',
        density=True, label=f'β_Mn = {beta_mean[bMn_idx]:.0f} ± {beta_std[bMn_idx]:.0f}')
ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Coefficient value (MPa·µm¹/² per unit fraction)')
ax.set_ylabel('Density')
ax.set_title('(c) Posterior of k_HP composition coefficients')
ax.legend(fontsize=9)

# Add P(>0) annotations
p_bV_pos = (beta_post[:, bV_idx] > 0).mean()
p_bMn_neg = (beta_post[:, bMn_idx] < 0).mean()
ax.text(0.98, 0.95, f'P(β_V > 0) = {p_bV_pos:.2f}\nP(β_Mn < 0) = {p_bMn_neg:.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/42_kHP_bayesian_composition.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 42_kHP_bayesian_composition.png")

# --- Plot 43: k_eff distribution + decomposition ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 43a: Histogram of k_eff
ax = axes[0]
ax.hist(k_eff, bins=25, color='steelblue', edgecolor='k', alpha=0.7, density=True)
ax.axvline(x=k_global, color='red', linestyle='--', linewidth=2, label=f'k_global = {k_global:.0f}')
ax.axvline(x=k_eff.mean(), color='blue', linestyle='-', linewidth=2, label=f'k_eff mean = {k_eff.mean():.0f}')
ax.axvline(x=494, color='green', linestyle=':', linewidth=1.5, label='CoCrFeMnNi (494)')
ax.set_xlabel('k_eff (MPa·µm¹/²)')
ax.set_ylabel('Density')
ax.set_title('(a) Distribution of effective k_HP')
ax.legend(fontsize=8)

# 43b: k_eff vs grain size (is k_eff grain-size dependent? shouldn't be)
ax = axes[1]
ax.scatter(d, k_eff, c=V * 100, cmap='plasma', s=30, edgecolors='k', linewidth=0.3)
slope_d, intercept_d, r_d, p_d, _ = stats.linregress(d, k_eff)
ax.plot([d.min(), d.max()], [slope_d*d.min()+intercept_d, slope_d*d.max()+intercept_d],
        'r--', linewidth=1.5)
ax.set_xlabel('Grain size (µm)')
ax.set_ylabel('k_eff (MPa·µm¹/²)')
ax.set_title(f'(b) k_eff vs grain size (r={r_d:.2f}, p={p_d:.3f})')
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=V.min()*100, vmax=V.max()*100))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='V (at%)', shrink=0.8)

# 43c: Parity: full model with k(V,Mn) vs constant k
ax = axes[2]
y_pred_const_k = sigma0_comp + k_global * d_inv_sqrt
y_pred_var_k = X_bayes @ beta_mean

ax.scatter(y, y_pred_const_k, c='gray', s=25, alpha=0.5, label=f'Constant k (R²={1-np.sum((y-y_pred_const_k)**2)/np.sum((y-y.mean())**2):.3f})')
ax.scatter(y, y_pred_var_k, c='steelblue', s=25, alpha=0.7, label=f'k(V,Mn) (R²={1-np.sum((y-y_pred_var_k)**2)/np.sum((y-y.mean())**2):.3f})')
ax.plot([100, 600], [100, 600], 'k--', linewidth=0.8)
ax.set_xlabel('Experimental YS (MPa)')
ax.set_ylabel('Predicted YS (MPa)')
ax.set_title('(c) Parity: constant k vs k(V,Mn)')
ax.legend(fontsize=9)
ax.set_xlim(100, 600)
ax.set_ylim(100, 600)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/43_kHP_diagnostics.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 43_kHP_diagnostics.png")

# ============================================================
# 11. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: IS k_HP COMPOSITION-DEPENDENT?")
print("=" * 70)

print(f"""
  Two-stage analysis (σ₀ from M3, then k_eff = residual/d⁻¹/²):
    k_eff mean = {k_eff.mean():.0f} ± {k_eff.std():.0f} MPa·µm¹/² (CV = {k_eff.std()/k_eff.mean()*100:.0f}%)
    k_eff range = {k_eff.min():.0f} – {k_eff.max():.0f} MPa·µm¹/²

  Regression k_eff = k₀ + Σβᵢ·xᵢ:
    R² = {r2_k:.3f} (composition explains {r2_k*100:.0f}% of k_eff variance)
    F-test p = {p_f:.4f} {'(significant)' if p_f < 0.05 else '(not significant)'}

  Bayesian model comparison (PSIS-LOO):
""")

for name, row in comparison.iterrows():
    print(f"    {name}: elpd = {row['elpd_loo']:.1f}, Δ = {row['elpd_diff']:.1f}, weight = {row['weight']:.3f}")

print(f"""
  Bayesian β coefficients for k(comp):
    β_V  = {beta_mean[bV_idx]:+.0f} ± {beta_std[bV_idx]:.0f}  P(β_V > 0) = {(beta_post[:, bV_idx] > 0).mean():.2f}
    β_Mn = {beta_mean[bMn_idx]:+.0f} ± {beta_std[bMn_idx]:.0f}  P(β_Mn < 0) = {(beta_post[:, bMn_idx] < 0).mean():.2f}

  Verdict: {"k_HP shows statistically significant composition dependence" if p_f < 0.05 else "k_HP composition dependence is NOT statistically significant"}
  {"The data suggest weak trends (V increases k, Mn decreases k) but" if p_f > 0.01 else ""}
  {"the effect is modest relative to the scatter." if p_f > 0.01 else ""}
""")

print("=" * 70)
print("DONE — All plots saved to analysis_plots/")
print("=" * 70)
