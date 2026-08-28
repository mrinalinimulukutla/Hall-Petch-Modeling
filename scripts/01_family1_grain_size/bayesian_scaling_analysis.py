#!/usr/bin/env python3
"""
Full Bayesian Analysis of Grain-Size Scaling Laws
===================================================
Compares 8 grain-size scaling laws using PyMC MCMC sampling,
PSIS-LOO model comparison, and Bayesian Model Averaging.

Models: σ_y = σ₀ + k·f(d) + ε,  ε ~ N(0, σ²)
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
import pytensor.tensor as pt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("FULL BAYESIAN ANALYSIS OF GRAIN-SIZE SCALING LAWS")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df_ys = df.dropna(subset=['YS']).copy()
y = df_ys['YS'].values.astype(np.float64)
d = df_ys['GrainSize'].values.astype(np.float64)
n = len(y)
print(f"\nDataset: {n} alloys, YS range: {y.min():.0f}–{y.max():.0f} MPa")
print(f"Grain size range: {d.min():.0f}–{d.max():.0f} µm")

# ============================================================
# 2. DEFINE SCALING LAW FEATURES
# ============================================================
scaling_laws = {
    'Hall–Petch (d⁻¹ᐟ²)':       d ** (-0.5),
    'Dunstan–Bushby (d⁻¹)':      d ** (-1.0),
    'Baldwin (d⁻¹ᐟ³)':           d ** (-1.0/3.0),
    'Intermediate (d⁻²ᐟ³)':      d ** (-2.0/3.0),
    'Crit. thickness (ln d/d)':   np.log(d) / d,
    'Logarithmic (ln d)':         np.log(d),
}

# ============================================================
# 3. FIT PYMC MODELS
# ============================================================
print("\n" + "=" * 70)
print("FITTING BAYESIAN MODELS (PyMC MCMC)")
print("=" * 70)

traces = {}
models = {}

# --- 3a. Two-parameter linear models (6 models) ---
for name, f_d in scaling_laws.items():
    print(f"\n  Fitting: {name} ...")
    with pm.Model() as model:
        sigma0 = pm.Normal('sigma0', mu=200, sigma=200)
        k = pm.Normal('k', mu=0, sigma=1000)
        sigma = pm.HalfCauchy('sigma', beta=50)
        mu = sigma0 + k * f_d
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(4000, tune=2000, cores=2, chains=2,
                          random_seed=42, progressbar=True,
                          return_inferencedata=True)
        # Compute log-likelihood for LOO
        pm.compute_log_likelihood(trace, model=model)
    traces[name] = trace
    models[name] = model
    # Quick diagnostics
    rhat_max = max(float(az.rhat(trace).max()[v]) for v in ['sigma0', 'k', 'sigma'])
    print(f"    R-hat max: {rhat_max:.4f}")

# --- 3b. Composite model: σ₀ + k₁·d⁻¹/² + k₂·d⁻¹ ---
print(f"\n  Fitting: Composite (d⁻¹ᐟ² + d⁻¹) ...")
f_hp = d ** (-0.5)
f_db = d ** (-1.0)
with pm.Model() as model_comp:
    sigma0 = pm.Normal('sigma0', mu=200, sigma=200)
    k1 = pm.Normal('k1', mu=0, sigma=1000)
    k2 = pm.Normal('k2', mu=0, sigma=1000)
    sigma = pm.HalfCauchy('sigma', beta=50)
    mu = sigma0 + k1 * f_hp + k2 * f_db
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    trace_comp = pm.sample(4000, tune=2000, cores=2, chains=2,
                           random_seed=42, progressbar=True,
                           return_inferencedata=True)
    pm.compute_log_likelihood(trace_comp, model=model_comp)
traces['Composite (d⁻¹ᐟ² + d⁻¹)'] = trace_comp
models['Composite (d⁻¹ᐟ² + d⁻¹)'] = model_comp
rhat_max = max(float(az.rhat(trace_comp).max()[v]) for v in ['sigma0', 'k1', 'k2', 'sigma'])
print(f"    R-hat max: {rhat_max:.4f}")

# --- 3c. Optimized exponent: σ₀ + k·d⁻ⁿ ---
print(f"\n  Fitting: Optimized exponent (d⁻ⁿ) ...")
with pm.Model() as model_opt:
    sigma0 = pm.Normal('sigma0', mu=200, sigma=200)
    k = pm.Normal('k', mu=0, sigma=1000)
    n_exp = pm.Uniform('n_exp', lower=0.1, upper=2.0)
    sigma = pm.HalfCauchy('sigma', beta=50)
    mu = sigma0 + k * pt.power(d, -n_exp)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    trace_opt = pm.sample(4000, tune=2000, cores=2, chains=2,
                          random_seed=42, progressbar=True,
                          return_inferencedata=True,
                          target_accept=0.9)  # higher for nonlinear
    pm.compute_log_likelihood(trace_opt, model=model_opt)
traces['Optimized exponent (d⁻ⁿ)'] = trace_opt
models['Optimized exponent (d⁻ⁿ)'] = model_opt
rhat_max = max(float(az.rhat(trace_opt).max()[v]) for v in ['sigma0', 'k', 'n_exp', 'sigma'])
print(f"    R-hat max: {rhat_max:.4f}")

# ============================================================
# 4. PSIS-LOO MODEL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("PSIS-LOO MODEL COMPARISON")
print("=" * 70)

comparison = az.compare(traces, ic='loo', method='stacking', scale='log')
print("\n" + str(comparison))

# Save comparison table
comparison.to_csv(f'{RESULTS_DIR}/bayesian_model_comparison.csv')
print(f"\nSaved comparison table to bayesian_model_comparison.csv")

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
# 5. PARAMETER SUMMARIES
# ============================================================
print("\n" + "=" * 70)
print("PARAMETER POSTERIORS (94% HDI)")
print("=" * 70)

for name, trace in traces.items():
    print(f"\n  {name}:")
    summary = az.summary(trace, hdi_prob=0.94,
                         var_names=[v for v in trace.posterior.data_vars if v != 'y_obs'])
    for idx, row in summary.iterrows():
        print(f"    {idx:>10s}: {row['mean']:8.1f}  [{row['hdi_3%']:8.1f}, {row['hdi_97%']:8.1f}]  "
              f"R-hat={row['r_hat']:.3f}  ESS={row['ess_bulk']:.0f}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# --- Plot 30: Model comparison (ΔLOO) ---
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(comparison, ax=ax, textsize=11)
ax.set_title('Bayesian Model Comparison (PSIS-LOO)', fontsize=14, fontweight='bold')
ax.set_xlabel('ELPD (LOO)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/30_bayesian_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 30_bayesian_model_comparison.png")

# --- Plot 31: Parameter posteriors (forest plot) ---
# Collect σ₀ and k posteriors for the 6 single-feature models
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# σ₀ posteriors
sigma0_data = {}
for name in scaling_laws.keys():
    sigma0_data[name] = traces[name].posterior['sigma0'].values.flatten()

ax = axes[0]
positions = range(len(sigma0_data))
labels = list(sigma0_data.keys())
for i, (name, vals) in enumerate(sigma0_data.items()):
    mean = np.mean(vals)
    hdi = az.hdi(vals, hdi_prob=0.94)
    ax.errorbar(mean, i, xerr=[[mean - hdi[0]], [hdi[1] - mean]],
                fmt='o', markersize=8, capsize=5, linewidth=2, color=f'C{i}')
ax.set_yticks(list(positions))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('σ₀ (MPa)', fontsize=12)
ax.set_title('Posterior: Friction Stress σ₀', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='x')

# k posteriors
k_data = {}
for name in scaling_laws.keys():
    k_data[name] = traces[name].posterior['k'].values.flatten()

ax = axes[1]
for i, (name, vals) in enumerate(k_data.items()):
    mean = np.mean(vals)
    hdi = az.hdi(vals, hdi_prob=0.94)
    ax.errorbar(mean, i, xerr=[[mean - hdi[0]], [hdi[1] - mean]],
                fmt='o', markersize=8, capsize=5, linewidth=2, color=f'C{i}')
ax.set_yticks(list(range(len(k_data))))
ax.set_yticklabels(list(k_data.keys()), fontsize=10)
ax.set_xlabel('k (scaling coefficient)', fontsize=12)
ax.set_title('Posterior: Scaling Coefficient k', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Parameter Posteriors (94% HDI)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/31_bayesian_posteriors.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 31_bayesian_posteriors.png")

# --- Plot 32: Posterior predictive checks (top 4 models) ---
top4_names = list(comparison.index[:4])
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
d_grid = np.linspace(d.min(), d.max(), 200)

for idx, name in enumerate(top4_names):
    ax = axes[idx // 2, idx % 2]
    trace = traces[name]

    # Get posterior samples
    sigma0_samples = trace.posterior['sigma0'].values.flatten()
    sigma_samples = trace.posterior['sigma'].values.flatten()

    if name == 'Composite (d⁻¹ᐟ² + d⁻¹)':
        k1_samples = trace.posterior['k1'].values.flatten()
        k2_samples = trace.posterior['k2'].values.flatten()
        # Draw 200 posterior curves
        n_draw = 200
        idx_draw = np.random.choice(len(sigma0_samples), n_draw, replace=False)
        for j in idx_draw:
            mu_j = sigma0_samples[j] + k1_samples[j] * d_grid**(-0.5) + k2_samples[j] * d_grid**(-1)
            ax.plot(d_grid, mu_j, color='steelblue', alpha=0.02, linewidth=0.5)
        # Mean prediction
        mu_mean = np.mean(sigma0_samples) + np.mean(k1_samples) * d_grid**(-0.5) + np.mean(k2_samples) * d_grid**(-1)
    elif name == 'Optimized exponent (d⁻ⁿ)':
        k_samples = trace.posterior['k'].values.flatten()
        n_samples = trace.posterior['n_exp'].values.flatten()
        n_draw = 200
        idx_draw = np.random.choice(len(sigma0_samples), n_draw, replace=False)
        for j in idx_draw:
            mu_j = sigma0_samples[j] + k_samples[j] * d_grid**(-n_samples[j])
            ax.plot(d_grid, mu_j, color='steelblue', alpha=0.02, linewidth=0.5)
        mu_mean = np.mean(sigma0_samples) + np.mean(k_samples) * d_grid**(-np.mean(n_samples))
    else:
        k_samples = trace.posterior['k'].values.flatten()
        # Determine f(d) for this model
        if 'd⁻¹ᐟ²' in name:
            f_grid = d_grid ** (-0.5)
        elif 'd⁻¹' in name and 'Composite' not in name:
            f_grid = d_grid ** (-1.0)
        elif 'd⁻¹ᐟ³' in name:
            f_grid = d_grid ** (-1.0/3.0)
        elif 'd⁻²ᐟ³' in name:
            f_grid = d_grid ** (-2.0/3.0)
        elif 'ln d/d' in name or 'ln(d)/d' in name or 'Crit' in name:
            f_grid = np.log(d_grid) / d_grid
        elif 'ln d' in name or 'ln(d)' in name or 'Logarithmic' in name:
            f_grid = np.log(d_grid)
        else:
            f_grid = d_grid ** (-0.5)  # fallback

        n_draw = 200
        idx_draw = np.random.choice(len(sigma0_samples), n_draw, replace=False)
        for j in idx_draw:
            mu_j = sigma0_samples[j] + k_samples[j] * f_grid
            ax.plot(d_grid, mu_j, color='steelblue', alpha=0.02, linewidth=0.5)
        mu_mean = np.mean(sigma0_samples) + np.mean(k_samples) * f_grid

    ax.plot(d_grid, mu_mean, color='navy', linewidth=2, label='Posterior mean')
    ax.scatter(d, y, c='orangered', s=30, alpha=0.7, edgecolors='k', linewidth=0.3,
               zorder=5, label='Data')
    ax.set_xlabel('Grain size d (µm)', fontsize=11)
    ax.set_ylabel('Yield Strength (MPa)', fontsize=11)
    # Get weight from comparison
    if name in comparison.index:
        w = comparison.loc[name, 'weight']
        ax.set_title(f'{name}\n(weight = {w:.3f})', fontsize=12, fontweight='bold')
    else:
        ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Checks (200 posterior draws)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/32_bayesian_ppc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 32_bayesian_ppc.png")

# --- Plot 33: BMA prediction ---
fig, ax = plt.subplots(figsize=(10, 7))
d_grid = np.linspace(d.min(), d.max(), 200)

# BMA: weighted average of posterior predictive means
bma_curves = []
weights = comparison['weight'].to_dict()

n_bma_samples = 500
bma_predictions = np.zeros((n_bma_samples, len(d_grid)))

for name, trace in traces.items():
    w = weights.get(name, 0)
    if w < 0.001:
        continue
    sigma0_s = trace.posterior['sigma0'].values.flatten()
    n_total = len(sigma0_s)
    n_from_this = max(1, int(n_bma_samples * w))
    idx_draw = np.random.choice(n_total, n_from_this, replace=False)

    for j_local, j in enumerate(idx_draw):
        if 'Composite' in name:
            k1_s = trace.posterior['k1'].values.flatten()[j]
            k2_s = trace.posterior['k2'].values.flatten()[j]
            mu_j = sigma0_s[j] + k1_s * d_grid**(-0.5) + k2_s * d_grid**(-1)
        elif 'Optimized' in name:
            k_s = trace.posterior['k'].values.flatten()[j]
            n_s = trace.posterior['n_exp'].values.flatten()[j]
            mu_j = sigma0_s[j] + k_s * d_grid**(-n_s)
        else:
            k_s = trace.posterior['k'].values.flatten()[j]
            if 'd⁻¹ᐟ²' in name:
                f_g = d_grid ** (-0.5)
            elif 'd⁻¹' in name:
                f_g = d_grid ** (-1.0)
            elif 'd⁻¹ᐟ³' in name:
                f_g = d_grid ** (-1.0/3.0)
            elif 'd⁻²ᐟ³' in name:
                f_g = d_grid ** (-2.0/3.0)
            elif 'Crit' in name:
                f_g = np.log(d_grid) / d_grid
            elif 'Logarithmic' in name:
                f_g = np.log(d_grid)
            else:
                f_g = d_grid ** (-0.5)
            mu_j = sigma0_s[j] + k_s * f_g
        bma_curves.append(mu_j)

bma_curves = np.array(bma_curves)
bma_mean = np.mean(bma_curves, axis=0)
bma_lo = np.percentile(bma_curves, 3, axis=0)
bma_hi = np.percentile(bma_curves, 97, axis=0)

ax.fill_between(d_grid, bma_lo, bma_hi, color='steelblue', alpha=0.25, label='94% credible band')
ax.plot(d_grid, bma_mean, color='navy', linewidth=2.5, label='BMA mean')
ax.scatter(d, y, c='orangered', s=40, alpha=0.8, edgecolors='k', linewidth=0.4,
           zorder=5, label='Data')
ax.set_xlabel('Grain size d (µm)', fontsize=13)
ax.set_ylabel('Yield Strength (MPa)', fontsize=13)
ax.set_title('Bayesian Model Averaging: Weighted Prediction', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/33_bayesian_bma.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 33_bayesian_bma.png")

# --- Plot 34: Optimized exponent posterior ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

n_exp_samples = traces['Optimized exponent (d⁻ⁿ)'].posterior['n_exp'].values.flatten()

ax = axes[0]
ax.hist(n_exp_samples, bins=60, density=True, color='steelblue', alpha=0.7, edgecolor='navy')
ax.axvline(x=0.5, color='red', linewidth=2, linestyle='--', label='Classical HP (n=0.5)')
hdi = az.hdi(n_exp_samples, hdi_prob=0.94)
ax.axvspan(hdi[0], hdi[1], alpha=0.15, color='orange', label=f'94% HDI [{hdi[0]:.3f}, {hdi[1]:.3f}]')
ax.set_xlabel('Exponent n', fontsize=12)
ax.set_ylabel('Posterior density', fontsize=12)
ax.set_title('Posterior of Grain-Size Exponent n', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# n vs k joint posterior
k_opt_samples = traces['Optimized exponent (d⁻ⁿ)'].posterior['k'].values.flatten()
ax = axes[1]
ax.scatter(n_exp_samples[::4], k_opt_samples[::4], alpha=0.1, s=3, c='steelblue')
ax.set_xlabel('Exponent n', fontsize=12)
ax.set_ylabel('Coefficient k', fontsize=12)
ax.set_title('Joint Posterior: n vs k', fontsize=13, fontweight='bold')
ax.axvline(x=0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# n vs σ₀ joint posterior
sigma0_opt = traces['Optimized exponent (d⁻ⁿ)'].posterior['sigma0'].values.flatten()
ax = axes[2]
ax.scatter(n_exp_samples[::4], sigma0_opt[::4], alpha=0.1, s=3, c='steelblue')
ax.set_xlabel('Exponent n', fontsize=12)
ax.set_ylabel('σ₀ (MPa)', fontsize=12)
ax.set_title('Joint Posterior: n vs σ₀', fontsize=13, fontweight='bold')
ax.axvline(x=0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.suptitle('Optimized Exponent Model: d⁻ⁿ', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/34_bayesian_exponent.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 34_bayesian_exponent.png")

# --- Plot 35: Stacking weights bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))
names = list(comparison.index)
wts = [comparison.loc[n, 'weight'] for n in names]
colors = ['#009E73' if w > 0.1 else '#E69F00' if w > 0.01 else '#D55E00' for w in wts]
bars = ax.barh(range(len(names)), wts, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('Stacking Weight (Posterior Model Probability)', fontsize=12)
ax.set_title('Bayesian Model Weights (Stacking)', fontsize=14, fontweight='bold')
# Annotate
for i, w in enumerate(wts):
    if w > 0.005:
        ax.text(w + 0.01, i, f'{w:.3f}', va='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, max(wts) * 1.25)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/35_bayesian_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 35_bayesian_weights.png")

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("BAYESIAN ANALYSIS SUMMARY")
print("=" * 70)

print("\nModel Rankings (PSIS-LOO):")
print(f"{'Model':<35s} {'ELPD_LOO':>10s} {'ΔLOO':>8s} {'Weight':>8s} {'p_loo':>6s}")
print("-" * 70)
for name in comparison.index:
    row = comparison.loc[name]
    delta = row.get('d_loo', row.get('elpd_diff', 0))
    p = row.get('p_loo', 0)
    print(f"  {name:<33s} {row['elpd_loo']:>10.1f} {delta:>8.1f} {row['weight']:>8.3f} {p:>6.1f}")

print(f"\nOptimized exponent posterior:")
print(f"  n = {np.mean(n_exp_samples):.3f} ± {np.std(n_exp_samples):.3f}")
print(f"  94% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]")
print(f"  P(n < 0.5) = {np.mean(n_exp_samples < 0.5):.3f}")
print(f"  P(0.4 < n < 0.6) = {np.mean((n_exp_samples > 0.4) & (n_exp_samples < 0.6)):.3f}")

print("\n" + "=" * 70)
print("DONE — All plots saved to analysis_plots/")
print("=" * 70)
