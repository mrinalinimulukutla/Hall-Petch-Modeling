#!/usr/bin/env python3
"""
Composition-Dependent Hall–Petch Analysis
==========================================
Tests whether σ₀ (friction stress) and k_HP (Hall–Petch coefficient)
depend on composition, comparing models of increasing complexity
via both frequentist (LOO R², BIC) and Bayesian (PSIS-LOO) criteria.

All models are of the form:
    σ_y = σ₀(comp) + k(comp) · d⁻¹/² + ε

where σ₀(comp) and k(comp) are linear functions of selected
composition variables. Since all models are linear in parameters,
MCMC converges in seconds.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
print("COMPOSITION-DEPENDENT HALL–PETCH ANALYSIS")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df = df.dropna(subset=['YS'])
n = len(df)

y = df['YS'].values.astype(float)
d = df['GrainSize'].values.astype(float)
d_inv_sqrt = d ** -0.5

# Element fractions (7 independent — Ni is solvent, dropped)
elem_names = ['Al_frac', 'Co_frac', 'Cr_frac', 'Cu_frac', 'Fe_frac', 'Mn_frac', 'V_frac']
elem_short = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'V']
X_elem = df[elem_names].values.astype(float)  # (n, 7)

# Individual elements for simple models
V = df['V_frac'].values.astype(float)
Mn = df['Mn_frac'].values.astype(float)
Al = df['Al_frac'].values.astype(float)

# Physics descriptors
delta = df['delta'].values.astype(float)

print(f"\nDataset: {n} alloys, YS range: {y.min():.0f}–{y.max():.0f} MPa")
print(f"Grain size range: {d.min():.0f}–{d.max():.0f} µm")

# ============================================================
# 2. DEFINE MODELS
# ============================================================
# Each model: σ_y = X @ β + ε  (linear in parameters)
# We define each by its design matrix and interpretive labels.

ones = np.ones(n)

models_spec = {
    # --- GROUP A: Composition-dependent σ₀ only ---
    'M0: Baseline HP': {
        'X': np.column_stack([ones, d_inv_sqrt]),
        'labels': ['σ₀', 'k'],
        'group': 'A: σ₀ only',
        'desc': 'σ₀ + k·d⁻¹/²',
    },
    'M1: σ₀(V)': {
        'X': np.column_stack([ones, V, d_inv_sqrt]),
        'labels': ['σ₀₀', 'α_V', 'k'],
        'group': 'A: σ₀ only',
        'desc': '(σ₀₀ + α_V·V) + k·d⁻¹/²',
    },
    'M2: σ₀(V,Mn)': {
        'X': np.column_stack([ones, V, Mn, d_inv_sqrt]),
        'labels': ['σ₀₀', 'α_V', 'α_Mn', 'k'],
        'group': 'A: σ₀ only',
        'desc': '(σ₀₀ + α_V·V + α_Mn·Mn) + k·d⁻¹/²',
    },
    'M3: σ₀(all elem)': {
        'X': np.column_stack([ones, X_elem, d_inv_sqrt]),
        'labels': ['σ₀₀'] + [f'α_{e}' for e in elem_short] + ['k'],
        'group': 'A: σ₀ only',
        'desc': '(σ₀₀ + Σα_i·x_i) + k·d⁻¹/²',
    },

    # --- GROUP B: Composition-dependent k only ---
    'M4: k(V)': {
        'X': np.column_stack([ones, d_inv_sqrt, V * d_inv_sqrt]),
        'labels': ['σ₀', 'k₀', 'β_V'],
        'group': 'B: k only',
        'desc': 'σ₀ + (k₀ + β_V·V)·d⁻¹/²',
    },
    'M5: k(V,Mn)': {
        'X': np.column_stack([ones, d_inv_sqrt, V * d_inv_sqrt, Mn * d_inv_sqrt]),
        'labels': ['σ₀', 'k₀', 'β_V', 'β_Mn'],
        'group': 'B: k only',
        'desc': 'σ₀ + (k₀ + β_V·V + β_Mn·Mn)·d⁻¹/²',
    },
    'M6: k(all elem)': {
        'X': np.column_stack([ones, d_inv_sqrt] + [X_elem[:, i] * d_inv_sqrt for i in range(7)]),
        'labels': ['σ₀', 'k₀'] + [f'β_{e}' for e in elem_short],
        'group': 'B: k only',
        'desc': 'σ₀ + (k₀ + Σβ_i·x_i)·d⁻¹/²',
    },

    # --- GROUP C: Both composition-dependent ---
    'M7: σ₀(V)+k(V)': {
        'X': np.column_stack([ones, V, d_inv_sqrt, V * d_inv_sqrt]),
        'labels': ['σ₀₀', 'α_V', 'k₀', 'β_V'],
        'group': 'C: Both',
        'desc': '(σ₀₀+α_V·V) + (k₀+β_V·V)·d⁻¹/²',
    },
    'M8: σ₀(V,Mn)+k(V,Mn)': {
        'X': np.column_stack([ones, V, Mn, d_inv_sqrt, V * d_inv_sqrt, Mn * d_inv_sqrt]),
        'labels': ['σ₀₀', 'α_V', 'α_Mn', 'k₀', 'β_V', 'β_Mn'],
        'group': 'C: Both',
        'desc': '(σ₀₀+α_V·V+α_Mn·Mn) + (k₀+β_V·V+β_Mn·Mn)·d⁻¹/²',
    },
    'M9: σ₀(V,Mn,Al)+k(V,Mn)': {
        'X': np.column_stack([ones, V, Mn, Al, d_inv_sqrt, V * d_inv_sqrt, Mn * d_inv_sqrt]),
        'labels': ['σ₀₀', 'α_V', 'α_Mn', 'α_Al', 'k₀', 'β_V', 'β_Mn'],
        'group': 'C: Both',
        'desc': '(σ₀₀+α·V+α·Mn+α·Al) + (k₀+β·V+β·Mn)·d⁻¹/²',
    },
    'M10: σ₀(all)+k(all)': {
        'X': np.column_stack([ones, X_elem, d_inv_sqrt] + [X_elem[:, i] * d_inv_sqrt for i in range(7)]),
        'labels': ['σ₀₀'] + [f'α_{e}' for e in elem_short] + ['k₀'] + [f'β_{e}' for e in elem_short],
        'group': 'C: Both',
        'desc': '(σ₀₀+Σα·x) + (k₀+Σβ·x)·d⁻¹/²',
    },

    # --- GROUP D: Physics descriptors ---
    'M11: σ₀(δ)': {
        'X': np.column_stack([ones, delta, d_inv_sqrt]),
        'labels': ['σ₀₀', 'α_δ', 'k'],
        'group': 'D: Physics',
        'desc': '(σ₀₀ + α_δ·δ) + k·d⁻¹/²',
    },
    'M12: σ₀(δ)+k(V)': {
        'X': np.column_stack([ones, delta, d_inv_sqrt, V * d_inv_sqrt]),
        'labels': ['σ₀₀', 'α_δ', 'k₀', 'β_V'],
        'group': 'D: Physics',
        'desc': '(σ₀₀+α_δ·δ) + (k₀+β_V·V)·d⁻¹/²',
    },
}

# ============================================================
# 3. OLS ANALYSIS (quick frequentist results)
# ============================================================
print("\n" + "=" * 70)
print("OLS ANALYSIS")
print("=" * 70)

ols_results = {}

for name, spec in models_spec.items():
    X = spec['X']
    # k_params = number of regression coefficients + 1 for the MLE noise
    # variance sigma^2.  BIC convention: ALWAYS count sigma^2 as an estimated
    # parameter (consistent across all scripts).  Note that
    # grain_size_scaling_analysis.py currently does NOT include +1 for sigma^2
    # in its k_params; that script should be updated to match this convention.
    k_params = X.shape[1] + 1  # +1 for noise variance (sigma^2)

    # OLS fit
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta_hat
    resid = y - y_pred

    # Train R²
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_train = 1 - ss_res / ss_tot

    # LOO R² via hat matrix (analytical, exact)
    H = X @ np.linalg.solve(X.T @ X, X.T)
    h_ii = np.diag(H)
    loo_resid = resid / (1 - h_ii)
    ss_loo = np.sum(loo_resid ** 2)
    r2_loo = 1 - ss_loo / ss_tot

    # BIC
    bic = n * np.log(ss_res / n) + k_params * np.log(n)

    # LOO RMSE
    loo_rmse = np.sqrt(ss_loo / n)

    ols_results[name] = {
        'beta': beta_hat,
        'r2_train': r2_train,
        'r2_loo': r2_loo,
        'loo_rmse': loo_rmse,
        'bic': bic,
        'k_params': k_params,
        'y_pred': y_pred,
        'loo_pred': y - loo_resid,  # LOO predictions
    }

# Print OLS results
print(f"\n{'Model':<30s} {'k':>3s} {'Train R²':>9s} {'LOO R²':>8s} {'RMSE':>6s} {'BIC':>8s} {'ΔBIC':>6s}")
print("-" * 80)

bic_min = min(r['bic'] for r in ols_results.values())
for name in models_spec:
    r = ols_results[name]
    dbic = r['bic'] - bic_min
    print(f"  {name:<28s} {r['k_params']:>3d} {r['r2_train']:>9.3f} {r['r2_loo']:>8.3f} "
          f"{r['loo_rmse']:>6.1f} {r['bic']:>8.1f} {dbic:>6.1f}")

# ============================================================
# 4. BAYESIAN ANALYSIS (PyMC MCMC + PSIS-LOO)
# ============================================================
print("\n" + "=" * 70)
print("BAYESIAN ANALYSIS (PyMC MCMC)")
print("=" * 70)

traces = {}
pymc_models = {}

for name, spec in models_spec.items():
    X = spec['X']
    n_feat = X.shape[1]
    print(f"\n  Fitting: {name} ({n_feat} coefficients + σ) ...")

    with pm.Model() as model:
        beta = pm.Normal('beta', mu=0, sigma=1000, shape=n_feat)
        sigma = pm.HalfCauchy('sigma', beta=50)
        mu = pm.math.dot(X, beta)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(4000, tune=2000, cores=2, chains=2,
                          random_seed=42, progressbar=True,
                          return_inferencedata=True)
        pm.compute_log_likelihood(trace, model=model)

    traces[name] = trace
    pymc_models[name] = model

    # Quick convergence check
    rhat_max = float(az.rhat(trace)['beta'].max())
    ess_min = float(az.ess(trace)['beta'].min())
    print(f"    R-hat max: {rhat_max:.4f}, ESS min: {ess_min:.0f}")

# ============================================================
# 5. PSIS-LOO COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("PSIS-LOO MODEL COMPARISON")
print("=" * 70)

comparison = az.compare(traces, ic='loo', method='stacking', scale='log')
print("\n" + str(comparison))

# Save comparison table
comparison.to_csv(f'{RESULTS_DIR}/comp_hp_model_comparison.csv')

# Check PSIS-LOO Pareto-k diagnostics
print("\n--- Pareto-k Diagnostics ---")
for model_name, trace in traces.items():
    loo_result = az.loo(trace, pointwise=True)
    pareto_k = loo_result.pareto_k.values
    n_bad = np.sum(pareto_k > 0.7)
    n_marginal = np.sum((pareto_k > 0.5) & (pareto_k <= 0.7))
    print(f"  {model_name}: {n_bad} bad (k>0.7), {n_marginal} marginal (0.5<k≤0.7), max k={pareto_k.max():.3f}")
    if n_bad > 0:
        print(f"    WARNING: {n_bad}/{len(pareto_k)} observations have unreliable PSIS-LOO estimates")

# ============================================================
# 6. EXTRACT KEY PARAMETERS FOR INTERPRETATION
# ============================================================
print("\n" + "=" * 70)
print("PARAMETER INTERPRETATION")
print("=" * 70)

# For selected models, show the physical parameters
for name in ['M0: Baseline HP', 'M8: σ₀(V,Mn)+k(V,Mn)', 'M10: σ₀(all)+k(all)']:
    spec = models_spec[name]
    trace = traces[name]
    labels = spec['labels']
    beta_post = trace.posterior['beta'].values  # (chains, draws, n_feat)
    beta_mean = beta_post.mean(axis=(0, 1))
    beta_std = beta_post.std(axis=(0, 1))

    print(f"\n  {name}:")
    print(f"    {spec['desc']}")
    for i, lbl in enumerate(labels):
        hdi = az.hdi(trace, var_names=['beta'])['beta'].values[i]
        print(f"    {lbl:>8s}: {beta_mean[i]:>9.1f} ± {beta_std[i]:>6.1f}  "
              f"[{hdi[0]:>8.1f}, {hdi[1]:>8.1f}]")
    sigma_mean = trace.posterior['sigma'].values.mean()
    print(f"    {'σ':>8s}: {sigma_mean:>9.1f}")

# ============================================================
# 7. COMPOSITION-DEPENDENT k_HP FOR BEST MODEL
# ============================================================
# For M8, compute k_HP(comp) = k₀ + β_V·V + β_Mn·Mn for each alloy
print("\n" + "=" * 70)
print("COMPOSITION-DEPENDENT k_HP (M8)")
print("=" * 70)

spec_m8 = models_spec['M8: σ₀(V,Mn)+k(V,Mn)']
beta_m8 = traces['M8: σ₀(V,Mn)+k(V,Mn)'].posterior['beta'].values.mean(axis=(0, 1))
labels_m8 = spec_m8['labels']
# labels: ['σ₀₀', 'α_V', 'α_Mn', 'k₀', 'β_V', 'β_Mn']

k0_idx = labels_m8.index('k₀')
bV_idx = labels_m8.index('β_V')
bMn_idx = labels_m8.index('β_Mn')

k_hp_per_alloy = beta_m8[k0_idx] + beta_m8[bV_idx] * V + beta_m8[bMn_idx] * Mn
sigma0_per_alloy = beta_m8[0] + beta_m8[1] * V + beta_m8[2] * Mn

print(f"\n  k₀ (base):        {beta_m8[k0_idx]:.0f} MPa·µm¹/²")
print(f"  β_V (V effect):   {beta_m8[bV_idx]:.0f} MPa·µm¹/²")
print(f"  β_Mn (Mn effect): {beta_m8[bMn_idx]:.0f} MPa·µm¹/²")
print(f"\n  k_HP range across alloys: {k_hp_per_alloy.min():.0f} – {k_hp_per_alloy.max():.0f} MPa·µm¹/²")
print(f"  σ₀ range across alloys:  {sigma0_per_alloy.min():.0f} – {sigma0_per_alloy.max():.0f} MPa")

# ============================================================
# 8. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# --- Plot 36: Model comparison bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Sort by LOO R²
sorted_names = sorted(ols_results.keys(), key=lambda x: ols_results[x]['r2_loo'], reverse=True)
colors_group = {'A: σ₀ only': '#4CAF50', 'B: k only': '#2196F3',
                'C: Both': '#FF9800', 'D: Physics': '#9C27B0'}

# 36a: LOO R²
r2_vals = [ols_results[n]['r2_loo'] for n in sorted_names]
bar_colors = [colors_group[models_spec[n]['group']] for n in sorted_names]
short_names = [n.split(': ', 1)[1] for n in sorted_names]
bars = axes[0].barh(range(len(sorted_names)), r2_vals, color=bar_colors, edgecolor='k', linewidth=0.5)
axes[0].set_yticks(range(len(sorted_names)))
axes[0].set_yticklabels(short_names, fontsize=9)
axes[0].set_xlabel('LOO R²')
axes[0].set_title('(a) LOO R² (higher = better)')
axes[0].invert_yaxis()
# Add value labels
for i, v in enumerate(r2_vals):
    axes[0].text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=8)

# 36b: ΔBIC
bic_vals = [ols_results[n]['bic'] - bic_min for n in sorted_names]
axes[1].barh(range(len(sorted_names)), bic_vals, color=bar_colors, edgecolor='k', linewidth=0.5)
axes[1].set_yticks(range(len(sorted_names)))
axes[1].set_yticklabels(short_names, fontsize=9)
axes[1].set_xlabel('ΔBIC')
axes[1].set_title('(b) ΔBIC (lower = better)')
axes[1].invert_yaxis()
for i, v in enumerate(bic_vals):
    axes[1].text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=8)

# 36c: ΔLOO (Bayesian)
comp_sorted = comparison.loc[[n for n in sorted_names if n in comparison.index]]
dloo_vals = []
for nm in sorted_names:
    if nm in comparison.index:
        dloo_vals.append(float(comparison.loc[nm, 'elpd_diff']))
    else:
        dloo_vals.append(0)
axes[2].barh(range(len(sorted_names)), [-v for v in dloo_vals], color=bar_colors, edgecolor='k', linewidth=0.5)
axes[2].set_yticks(range(len(sorted_names)))
axes[2].set_yticklabels(short_names, fontsize=9)
axes[2].set_xlabel('ΔLOO (elpd difference from best)')
axes[2].set_title('(c) PSIS-LOO (lower = better)')
axes[2].invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, edgecolor='k', label=g) for g, c in colors_group.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, frameon=True)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f'{PLOT_DIR}/36_comp_hp_model_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 36_comp_hp_model_comparison.png")

# --- Plot 37: Parity plots for top 4 models ---
top4 = sorted_names[:4]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, name in zip(axes, top4):
    y_loo = ols_results[name]['loo_pred']
    r2 = ols_results[name]['r2_loo']
    rmse = ols_results[name]['loo_rmse']
    ax.scatter(y, y_loo, c=V, cmap='plasma', s=30, edgecolors='k', linewidth=0.3, alpha=0.8)
    ax.plot([100, 600], [100, 600], 'k--', linewidth=0.8)
    ax.set_xlabel('Experimental YS (MPa)')
    ax.set_ylabel('LOO Predicted YS (MPa)')
    short = name.split(': ', 1)[1]
    ax.set_title(f'{short}\nR²={r2:.3f}, RMSE={rmse:.1f}', fontsize=10)
    ax.set_xlim(100, 600)
    ax.set_ylim(100, 600)
    ax.set_aspect('equal')
# Colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=V.min(), vmax=V.max()))
sm.set_array([])
fig.colorbar(sm, ax=axes[-1], label='V fraction', shrink=0.8)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/37_comp_hp_parity.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 37_comp_hp_parity.png")

# --- Plot 38: Coefficient visualization for M8 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 38a: σ₀(comp) vs V and Mn
ax = axes[0]
sc = ax.scatter(V * 100, sigma0_per_alloy, c=Mn * 100, cmap='coolwarm', s=50,
                edgecolors='k', linewidth=0.3)
ax.set_xlabel('V content (at%)')
ax.set_ylabel('σ₀(comp) = σ₀₀ + α_V·V + α_Mn·Mn  (MPa)')
ax.set_title('(a) Composition-dependent friction stress')
fig.colorbar(sc, ax=ax, label='Mn content (at%)')

# 38b: k_HP(comp) vs V and Mn
ax = axes[1]
sc = ax.scatter(V * 100, k_hp_per_alloy, c=Mn * 100, cmap='coolwarm', s=50,
                edgecolors='k', linewidth=0.3)
ax.set_xlabel('V content (at%)')
ax.set_ylabel('k_HP(comp) = k₀ + β_V·V + β_Mn·Mn  (MPa·µm¹/²)')
ax.set_title('(b) Composition-dependent HP coefficient')
fig.colorbar(sc, ax=ax, label='Mn content (at%)')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/38_comp_hp_coefficients.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 38_comp_hp_coefficients.png")

# --- Plot 39: R² progression ---
fig, ax = plt.subplots(figsize=(10, 5))

# Plot R² vs number of parameters for each group
for group, color in colors_group.items():
    group_models = [(n, ols_results[n]) for n in models_spec if models_spec[n]['group'] == group]
    if not group_models:
        continue
    kp = [r['k_params'] for _, r in group_models]
    r2 = [r['r2_loo'] for _, r in group_models]
    shorts = [n.split(': ', 1)[1] for n, _ in group_models]
    ax.scatter(kp, r2, c=color, s=100, edgecolors='k', linewidth=0.5, zorder=5, label=group)
    for k, r, s in zip(kp, r2, shorts):
        ax.annotate(s, (k, r), textcoords="offset points", xytext=(5, 5), fontsize=7)

ax.set_xlabel('Number of parameters (incl. σ)')
ax.set_ylabel('LOO R²')
ax.set_title('LOO R² vs Model Complexity')
ax.legend(loc='lower right', fontsize=9)
ax.axhline(y=ols_results['M0: Baseline HP']['r2_loo'], color='gray', linestyle=':', linewidth=0.8,
           label='Baseline HP')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/39_comp_hp_r2_progression.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 39_comp_hp_r2_progression.png")

# --- Plot 40: Posterior predictive for best model ---
best_name = sorted_names[0]
best_spec = models_spec[best_name]
best_trace = traces[best_name]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 40a: Predicted vs observed with uncertainty
beta_samples = best_trace.posterior['beta'].values.reshape(-1, best_spec['X'].shape[1])
sigma_samples = best_trace.posterior['sigma'].values.flatten()

# Posterior predictive mean and CI
y_pred_samples = beta_samples @ best_spec['X'].T  # (n_samples, n_alloys)
y_pred_mean = y_pred_samples.mean(axis=0)
y_pred_lo = np.percentile(y_pred_samples, 3, axis=0)
y_pred_hi = np.percentile(y_pred_samples, 97, axis=0)

# Sort by observed YS for plotting
sort_idx = np.argsort(y)
ax = axes[0]
ax.fill_between(range(n), y_pred_lo[sort_idx], y_pred_hi[sort_idx], alpha=0.3, color='steelblue',
                label='94% credible interval')
ax.plot(range(n), y_pred_mean[sort_idx], 'b-', linewidth=1, label='Posterior mean')
ax.scatter(range(n), y[sort_idx], c='red', s=15, zorder=5, label='Observed')
ax.set_xlabel('Alloy (sorted by YS)')
ax.set_ylabel('Yield Strength (MPa)')
short_best = best_name.split(': ', 1)[1]
ax.set_title(f'(a) {short_best}: predictions with 94% CI')
ax.legend(fontsize=9)

# 40b: Residuals
ax = axes[1]
resid_loo = y - ols_results[best_name]['loo_pred']
ax.scatter(ols_results[best_name]['loo_pred'], resid_loo, c=V, cmap='plasma',
           s=30, edgecolors='k', linewidth=0.3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
ax.set_xlabel('LOO Predicted YS (MPa)')
ax.set_ylabel('Residual (MPa)')
ax.set_title(f'(b) {short_best}: LOO residuals (colored by V)')
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=V.min(), vmax=V.max()))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='V fraction', shrink=0.8)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/40_comp_hp_best_model.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved 40_comp_hp_best_model.png")

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

baseline_r2 = ols_results['M0: Baseline HP']['r2_loo']
print(f"\n  Baseline HP LOO R²: {baseline_r2:.3f}")
print(f"\n  {'Model':<30s} {'LOO R²':>8s} {'ΔR²':>6s} {'ΔBIC':>6s}")
print("  " + "-" * 56)
for name in sorted_names:
    r = ols_results[name]
    dr2 = r['r2_loo'] - baseline_r2
    dbic = r['bic'] - bic_min
    marker = ' ★' if name == sorted_names[0] else ''
    print(f"  {name:<30s} {r['r2_loo']:>8.3f} {dr2:>+6.3f} {dbic:>6.1f}{marker}")

best_r2 = ols_results[sorted_names[0]]['r2_loo']
print(f"\n  Best model: {sorted_names[0]}")
print(f"  Improvement over baseline: ΔR² = {best_r2 - baseline_r2:+.3f} "
      f"({(best_r2 - baseline_r2) / baseline_r2 * 100:+.1f}%)")

# PSIS-LOO ranking
print(f"\n  PSIS-LOO top 5:")
for i, (name, row) in enumerate(comparison.head(5).iterrows()):
    print(f"    {i+1}. {name} (elpd={row['elpd_loo']:.1f}, Δ={row['elpd_diff']:.1f}, w={row['weight']:.3f})")

print(f"\n" + "=" * 70)
print("DONE — All plots saved to analysis_plots/")
print("=" * 70)
