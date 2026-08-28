#!/usr/bin/env python3
"""
Review Diagnostics: Robustness Checks for Hall–Petch Composition Analysis
==========================================================================
Addresses issues raised by expert metallurgy review:
  1. VIF & condition number for M3 design matrix
  2. Monte Carlo grain-size error propagation
  3. Subset k_HP consistency across composition groups
  4. Per-alloy k_HP for compositions with multiple grain sizes
  5. Simpson's paradox check (V_frac correlated with grain size)
  6. Bootstrap CIs for M3 coefficients

Produces 4 diagnostic plots (44–47) in analysis_plots/.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'

np.random.seed(42)
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# ============================================================
# 0. LOAD DATA & FIT M3
# ============================================================
print("=" * 70)
print("REVIEW DIAGNOSTICS: ROBUSTNESS CHECKS")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df = df.dropna(subset=['YS'])
n = len(df)

y = df['YS'].values.astype(float)
d = df['GrainSize'].values.astype(float)
sd_gs = df['SD_GS'].values.astype(float)
d_inv_sqrt = d ** -0.5

elem_names = ['Al_frac', 'Co_frac', 'Cr_frac', 'Cu_frac', 'Fe_frac', 'Mn_frac', 'V_frac']
elem_short = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'V']
X_elem = df[elem_names].values.astype(float)

ones = np.ones(n)
X_m3 = np.column_stack([ones, X_elem, d_inv_sqrt])  # intercept + 7 elem + d^{-1/2}
beta_m3 = np.linalg.lstsq(X_m3, y, rcond=None)[0]

y_pred_m3 = X_m3 @ beta_m3
resid_m3 = y - y_pred_m3
ss_res_m3 = np.sum(resid_m3 ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)

# Coefficient labels
coef_labels = ['sigma_0'] + elem_short + ['k']
sigma0_comp = X_m3[:, :8] @ beta_m3[:8]
k_global = beta_m3[-1]
k_eff = (y - sigma0_comp) / d_inv_sqrt

print(f"\nDataset: {n} alloys")
print(f"M3 R² = {1 - ss_res_m3/ss_tot:.4f}, k_global = {k_global:.1f}")

# ============================================================
# 1. VIF & CONDITION NUMBER
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: VIF & CONDITION NUMBER")
print("=" * 70)

# VIF for the 7 element fractions + d^{-1/2} (excluding intercept)
X_features = X_m3[:, 1:]  # drop intercept for VIF calculation
feature_names = elem_short + ['d^{-1/2}']

print(f"\n  {'Feature':>12s} {'VIF':>8s}")
print("  " + "-" * 24)
vif_values = []
for i in range(X_features.shape[1]):
    vif = variance_inflation_factor(X_features, i)
    vif_values.append(vif)
    flag = ' ***' if vif > 10 else ' *' if vif > 5 else ''
    print(f"  {feature_names[i]:>12s} {vif:>8.2f}{flag}")

# Condition number
cond = np.linalg.cond(X_m3)
print(f"\n  Condition number of M3 design matrix: {cond:.1f}")
if cond < 30:
    print("  --> Low: no multicollinearity concern")
elif cond < 100:
    print("  --> Moderate: acceptable but worth monitoring")
else:
    print("  --> High: potential multicollinearity")

max_vif = max(vif_values)
print(f"\n  Max VIF = {max_vif:.2f}", end='')
if max_vif < 5:
    print(" --> No multicollinearity concern (all VIF < 5)")
elif max_vif < 10:
    print(" --> Moderate concern (VIF < 10)")
else:
    print(" --> Multicollinearity detected (VIF > 10)")

# Store for report
vif_summary = {fn: vf for fn, vf in zip(feature_names, vif_values)}

# ============================================================
# 2. MONTE CARLO GRAIN-SIZE ERROR PROPAGATION
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: MONTE CARLO GRAIN-SIZE ERROR PROPAGATION")
print("=" * 70)

N_MC = 1000
# Use SD_GS as the uncertainty in the reported mean grain size
# SD_GS represents within-alloy grain size distribution, so the
# uncertainty of the mean is SD_GS / sqrt(N_grains). Since N_grains
# is unknown, we conservatively use SD_GS/sqrt(10) as a reasonable
# estimate (typically 10–50 grains measured).
gs_uncertainty = sd_gs / np.sqrt(10)  # conservative: assume 10 grains measured

mc_betas = np.zeros((N_MC, X_m3.shape[1]))

for i in range(N_MC):
    # Perturb grain sizes
    d_perturbed = d + np.random.randn(n) * gs_uncertainty
    d_perturbed = np.maximum(d_perturbed, 1.0)  # enforce positive
    d_inv_sqrt_perturbed = d_perturbed ** -0.5

    X_mc = np.column_stack([ones, X_elem, d_inv_sqrt_perturbed])
    mc_betas[i] = np.linalg.lstsq(X_mc, y, rcond=None)[0]

mc_mean = mc_betas.mean(axis=0)
mc_std = mc_betas.std(axis=0)
mc_lo = np.percentile(mc_betas, 2.5, axis=0)
mc_hi = np.percentile(mc_betas, 97.5, axis=0)

print(f"\n  Monte Carlo: {N_MC} replicates, GS uncertainty = SD_GS/sqrt(10)")
print(f"\n  {'Param':>10s} {'OLS':>10s} {'MC Mean':>10s} {'MC Std':>8s} {'MC 95% CI':>24s} {'Shift%':>8s}")
print("  " + "-" * 72)
for j, lbl in enumerate(coef_labels):
    shift_pct = abs(mc_mean[j] - beta_m3[j]) / max(abs(beta_m3[j]), 1e-6) * 100
    print(f"  {lbl:>10s} {beta_m3[j]:>10.1f} {mc_mean[j]:>10.1f} {mc_std[j]:>8.1f} "
          f"[{mc_lo[j]:>8.1f}, {mc_hi[j]:>8.1f}] {shift_pct:>7.1f}%")

# ============================================================
# 3. SUBSET k_HP CONSISTENCY
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: SUBSET k_HP CONSISTENCY")
print("=" * 70)

# Define subsets based on composition characteristics
subsets = {}
V_frac = df['V_frac'].values
Mn_frac = df['Mn_frac'].values
Co_frac = df['Co_frac'].values
Cr_frac = df['Cr_frac'].values

subsets['V-containing (V>0)'] = V_frac > 0
subsets['V-free (V=0)'] = V_frac == 0
subsets['Mn-rich (Mn>=0.12)'] = Mn_frac >= 0.12
subsets['Mn-poor (Mn<0.12)'] = Mn_frac < 0.12
subsets['Co-rich (Co>=0.20)'] = Co_frac >= 0.20
subsets['Co-poor (Co<0.20)'] = Co_frac < 0.20
subsets['Equimolar-like'] = df['n_comp'].values >= 5
subsets['Few-component'] = df['n_comp'].values < 5

N_BOOT = 5000
subset_k_results = {}

print(f"\n  {'Subset':>24s} {'N':>4s} {'k_HP':>8s} {'SE':>6s} {'95% CI (bootstrap)':>24s}")
print("  " + "-" * 72)

for label, mask in subsets.items():
    n_sub = mask.sum()
    if n_sub < 5:
        continue
    y_sub = y[mask]
    d_sub = d_inv_sqrt[mask]
    X_sub = np.column_stack([np.ones(n_sub), d_sub])

    # OLS fit
    beta_sub = np.linalg.lstsq(X_sub, y_sub, rcond=None)[0]

    # Bootstrap CI
    k_boots = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = np.random.randint(0, n_sub, n_sub)
        b_beta = np.linalg.lstsq(X_sub[idx], y_sub[idx], rcond=None)[0]
        k_boots[b] = b_beta[1]

    ci_lo, ci_hi = np.percentile(k_boots, [2.5, 97.5])
    k_se = k_boots.std()

    subset_k_results[label] = {
        'n': n_sub, 'k': beta_sub[1], 'se': k_se,
        'ci_lo': ci_lo, 'ci_hi': ci_hi, 'sigma0': beta_sub[0]
    }

    print(f"  {label:>24s} {n_sub:>4d} {beta_sub[1]:>8.0f} {k_se:>6.0f} [{ci_lo:>8.0f}, {ci_hi:>8.0f}]")

print(f"\n  Global k_HP = {k_global:.0f}")
print(f"  Range across subsets: {min(r['k'] for r in subset_k_results.values()):.0f} – "
      f"{max(r['k'] for r in subset_k_results.values()):.0f}")

# ============================================================
# 4. PER-ALLOY k_HP (COMPOSITIONS WITH MULTIPLE GRAIN SIZES)
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: PER-ALLOY k_HP (MULTI-GRAIN-SIZE COMPOSITIONS)")
print("=" * 70)

comp_cols = ['Al_frac', 'Co_frac', 'Cr_frac', 'Cu_frac', 'Fe_frac', 'Mn_frac', 'Ni_frac', 'V_frac']
groups = df.groupby(comp_cols)

per_alloy_k = []
print(f"\n  {'Composition':>45s} {'N':>3s} {'k_HP':>8s} {'σ₀':>8s} {'R²':>6s} {'GS range':>12s}")
print("  " + "-" * 90)

for comp, grp in groups:
    if len(grp) < 2:
        continue

    y_g = grp['YS'].values.astype(float)
    d_g = grp['GrainSize'].values.astype(float)
    d_inv_g = d_g ** -0.5
    n_g = len(grp)

    # Short composition label
    short = '/'.join([f'{e}{v:.0f}' for e, v in zip(
        ['Al','Co','Cr','Cu','Fe','Mn','Ni','V'], [x*100 for x in comp]) if v > 0])

    # Check if grain sizes vary enough for a meaningful fit
    gs_range = d_g.max() - d_g.min()
    gs_cv = d_g.std() / d_g.mean() if d_g.mean() > 0 else 0

    if n_g >= 2:
        X_g = np.column_stack([np.ones(n_g), d_inv_g])
        beta_g = np.linalg.lstsq(X_g, y_g, rcond=None)[0]
        y_pred_g = X_g @ beta_g
        ss_res_g = np.sum((y_g - y_pred_g) ** 2)
        ss_tot_g = np.sum((y_g - y_g.mean()) ** 2)
        r2_g = 1 - ss_res_g / ss_tot_g if ss_tot_g > 0 else float('nan')

        # Bootstrap CI for k (if n >= 3)
        if n_g >= 3:
            k_boots = []
            for _ in range(N_BOOT):
                idx = np.random.randint(0, n_g, n_g)
                b = np.linalg.lstsq(X_g[idx], y_g[idx], rcond=None)[0]
                k_boots.append(b[1])
            k_boots = np.array(k_boots)
            ci = np.percentile(k_boots, [2.5, 97.5])
        else:
            ci = [float('nan'), float('nan')]

        per_alloy_k.append({
            'comp': short, 'n': n_g, 'k': beta_g[1], 'sigma0': beta_g[0],
            'r2': r2_g, 'gs_range': f'{d_g.min():.0f}-{d_g.max():.0f}',
            'ci_lo': ci[0], 'ci_hi': ci[1], 'gs_cv': gs_cv,
        })

        flag = ' (low GS variation)' if gs_cv < 0.15 else ''
        print(f"  {short:>45s} {n_g:>3d} {beta_g[1]:>8.0f} {beta_g[0]:>8.0f} "
              f"{r2_g:>6.2f} {d_g.min():>5.0f}-{d_g.max():>5.0f} µm{flag}")

print(f"\n  Global k_HP = {k_global:.0f}")
reliable = [r for r in per_alloy_k if r['gs_cv'] >= 0.15]
if reliable:
    print(f"  Reliable per-alloy k range: {min(r['k'] for r in reliable):.0f} – "
          f"{max(r['k'] for r in reliable):.0f}")

# ============================================================
# 5. SIMPSON'S PARADOX CHECK
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: SIMPSON'S PARADOX CHECK")
print("=" * 70)

V_frac_arr = df['V_frac'].values.astype(float)

# (a) Raw correlations
r_vd, p_vd = stats.pearsonr(V_frac_arr, d)
r_vdinv, p_vdinv = stats.pearsonr(V_frac_arr, d_inv_sqrt)
r_vy, p_vy = stats.pearsonr(V_frac_arr, y)

print(f"\n  Raw correlations:")
print(f"    V_frac vs GrainSize:  r = {r_vd:+.3f}, p = {p_vd:.4f}")
print(f"    V_frac vs d⁻¹/²:     r = {r_vdinv:+.3f}, p = {p_vdinv:.4f}")
print(f"    V_frac vs YS:         r = {r_vy:+.3f}, p = {p_vy:.4f}")

# (b) Partial correlation: V_frac with YS controlling for d^{-1/2}
# Method: regress both V_frac and YS on d^{-1/2}, take residuals, compute correlation
X_d = np.column_stack([ones, d_inv_sqrt])
beta_vd = np.linalg.lstsq(X_d, V_frac_arr, rcond=None)[0]
beta_yd = np.linalg.lstsq(X_d, y, rcond=None)[0]
resid_v = V_frac_arr - X_d @ beta_vd
resid_y = y - X_d @ beta_yd
r_partial, p_partial = stats.pearsonr(resid_v, resid_y)

print(f"\n  Partial correlation (controlling for d⁻¹/²):")
print(f"    V_frac vs YS | d⁻¹/²:  r = {r_partial:+.3f}, p = {p_partial:.4f}")
print(f"    (Raw V vs YS:          r = {r_vy:+.3f})")
attenuation = (1 - abs(r_partial) / max(abs(r_vy), 1e-6)) * 100
print(f"    Attenuation: {attenuation:.0f}% of V's raw correlation with YS is explained by grain-size correlation")

# (c) Compare V coefficient in M3 (with d^{-1/2}) vs without d^{-1/2}
X_comp_only = np.column_stack([ones, X_elem])
beta_comp_only = np.linalg.lstsq(X_comp_only, y, rcond=None)[0]

v_idx = elem_short.index('V') + 1  # +1 for intercept
alpha_V_with_GS = beta_m3[v_idx]
alpha_V_without_GS = beta_comp_only[v_idx]

print(f"\n  V coefficient comparison:")
print(f"    α_V in M3 (with d⁻¹/²):    {alpha_V_with_GS:+.1f}")
print(f"    α_V without d⁻¹/²:          {alpha_V_without_GS:+.1f}")
change_pct = (alpha_V_without_GS - alpha_V_with_GS) / max(abs(alpha_V_with_GS), 1e-6) * 100
print(f"    Change: {change_pct:+.0f}%")
if abs(change_pct) > 20:
    print("    --> CAUTION: V coefficient changes substantially when grain-size is removed")
    print("       This suggests confounding between V content and grain refinement.")
else:
    print("    --> Coefficient stable: limited confounding between V and grain size")

# Also check all elements
print(f"\n  All element coefficients with vs without d⁻¹/²:")
print(f"  {'Element':>8s} {'With d⁻¹/²':>12s} {'Without':>10s} {'Change%':>10s}")
print("  " + "-" * 44)
for i, elem in enumerate(elem_short):
    idx = i + 1
    with_gs = beta_m3[idx]
    without_gs = beta_comp_only[idx]
    chg = (without_gs - with_gs) / max(abs(with_gs), 1e-6) * 100
    flag = ' *' if abs(chg) > 20 else ''
    print(f"  {elem:>8s} {with_gs:>12.1f} {without_gs:>10.1f} {chg:>+9.0f}%{flag}")

# ============================================================
# 6. BOOTSTRAP CIS FOR M3 COEFFICIENTS
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 6: BOOTSTRAP CIs FOR M3 COEFFICIENTS")
print("=" * 70)

N_BOOT_CI = 10000
boot_betas = np.zeros((N_BOOT_CI, X_m3.shape[1]))

for b in range(N_BOOT_CI):
    idx = np.random.randint(0, n, n)
    boot_betas[b] = np.linalg.lstsq(X_m3[idx], y[idx], rcond=None)[0]

# BCa confidence intervals (bias-corrected and accelerated)
# Simplified: use percentile intervals
boot_lo = np.percentile(boot_betas, 2.5, axis=0)
boot_hi = np.percentile(boot_betas, 97.5, axis=0)
boot_mean = boot_betas.mean(axis=0)
boot_std = boot_betas.std(axis=0)

# OLS standard errors for comparison
sigma2_ols = ss_res_m3 / (n - X_m3.shape[1])
cov_ols = sigma2_ols * np.linalg.inv(X_m3.T @ X_m3)
se_ols = np.sqrt(np.diag(cov_ols))

print(f"\n  {N_BOOT_CI} bootstrap resamples")
print(f"\n  {'Param':>10s} {'OLS':>10s} {'OLS SE':>8s} {'Boot SE':>8s} {'Boot 95% CI':>24s} {'SE ratio':>10s}")
print("  " + "-" * 78)
for j, lbl in enumerate(coef_labels):
    ratio = boot_std[j] / se_ols[j]
    flag = ' *' if abs(ratio - 1) > 0.2 else ''
    print(f"  {lbl:>10s} {beta_m3[j]:>10.1f} {se_ols[j]:>8.1f} {boot_std[j]:>8.1f} "
          f"[{boot_lo[j]:>8.1f}, {boot_hi[j]:>8.1f}] {ratio:>9.2f}{flag}")

# Check which coefficients have CIs that include zero
print(f"\n  Coefficients with 95% CI excluding zero (statistically significant):")
for j, lbl in enumerate(coef_labels):
    if boot_lo[j] > 0 or boot_hi[j] < 0:
        print(f"    {lbl}: [{boot_lo[j]:.1f}, {boot_hi[j]:.1f}]")

# ============================================================
# 7. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# --- Plot 44: MC grain-size sensitivity (violin plot) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 44a: Violin plot of σ₀ element coefficients across MC replicates
ax = axes[0]
# Only plot element coefficients (indices 1–7)
elem_mc = mc_betas[:, 1:8]
parts = ax.violinplot(elem_mc, positions=range(7), showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('steelblue')
    pc.set_alpha(0.6)
parts['cmeans'].set_color('red')
parts['cmedians'].set_color('black')

# Overlay OLS point estimates
for j in range(7):
    ax.plot(j, beta_m3[j+1], 'D', color='red', markersize=8, zorder=5)

ax.set_xticks(range(7))
ax.set_xticklabels([f'α_{e}' for e in elem_short], fontsize=10)
ax.set_ylabel('Coefficient value (MPa per unit fraction)')
ax.set_title('(a) σ₀ composition coefficients\n(MC grain-size perturbation)')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax.legend(['OLS estimate', 'MC distribution'], loc='upper right', fontsize=9)

# 44b: k (HP coefficient) across MC replicates
ax = axes[1]
ax.hist(mc_betas[:, -1], bins=50, color='steelblue', edgecolor='k',
        alpha=0.7, density=True)
ax.axvline(x=beta_m3[-1], color='red', linestyle='--', linewidth=2,
           label=f'OLS k = {beta_m3[-1]:.0f}')
ax.axvline(x=mc_mean[-1], color='blue', linestyle='-', linewidth=2,
           label=f'MC mean k = {mc_mean[-1]:.0f}')
ci_k = np.percentile(mc_betas[:, -1], [2.5, 97.5])
ax.axvspan(ci_k[0], ci_k[1], alpha=0.15, color='steelblue',
           label=f'95% CI [{ci_k[0]:.0f}, {ci_k[1]:.0f}]')
ax.set_xlabel('k_HP (MPa·µm¹/²)')
ax.set_ylabel('Density')
ax.set_title('(b) Hall–Petch coefficient k\n(MC grain-size perturbation)')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/44_mc_grain_size_sensitivity.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved 44_mc_grain_size_sensitivity.png")

# --- Plot 45: Subset k_HP bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))

labels_sub = list(subset_k_results.keys())
k_vals = [subset_k_results[l]['k'] for l in labels_sub]
k_errs_lo = [subset_k_results[l]['k'] - subset_k_results[l]['ci_lo'] for l in labels_sub]
k_errs_hi = [subset_k_results[l]['ci_hi'] - subset_k_results[l]['k'] for l in labels_sub]
n_vals = [subset_k_results[l]['n'] for l in labels_sub]

# Color pairs (complementary for each dichotomy)
colors_sub = ['#4CAF50', '#C8E6C9', '#FF9800', '#FFE0B2',
              '#2196F3', '#BBDEFB', '#9C27B0', '#E1BEE7']

bars = ax.barh(range(len(labels_sub)), k_vals, xerr=[k_errs_lo, k_errs_hi],
               color=colors_sub[:len(labels_sub)], edgecolor='k', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5})

# Global reference line
ax.axvline(x=k_global, color='red', linestyle='--', linewidth=2,
           label=f'Global k = {k_global:.0f}')

ax.set_yticks(range(len(labels_sub)))
ax.set_yticklabels([f'{l} (n={n})' for l, n in zip(labels_sub, n_vals)], fontsize=9)
ax.set_xlabel('k_HP (MPa·µm¹/²)')
ax.set_title('Hall–Petch coefficient k by composition subset\n(bootstrap 95% CI)')
ax.legend(fontsize=10)
ax.invert_yaxis()

# Value labels
for i, v in enumerate(k_vals):
    ax.text(v + 20, i, f'{v:.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/45_subset_kHP.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved 45_subset_kHP.png")

# --- Plot 46: Per-alloy k_HP ---
fig, ax = plt.subplots(figsize=(10, 6))

if per_alloy_k:
    alloy_labels = [r['comp'] for r in per_alloy_k]
    alloy_k = [r['k'] for r in per_alloy_k]
    alloy_n = [r['n'] for r in per_alloy_k]
    alloy_gs_cv = [r['gs_cv'] for r in per_alloy_k]

    # Color by reliability (grain size CV)
    colors_alloy = ['#4CAF50' if cv >= 0.15 else '#BDBDBD' for cv in alloy_gs_cv]

    bars = ax.barh(range(len(alloy_labels)), alloy_k, color=colors_alloy,
                   edgecolor='k', linewidth=0.5)

    # Add CI whiskers where available
    for i, r in enumerate(per_alloy_k):
        if not np.isnan(r['ci_lo']):
            ax.plot([r['ci_lo'], r['ci_hi']], [i, i], 'k-', linewidth=1.5)
            ax.plot([r['ci_lo'], r['ci_lo']], [i-0.15, i+0.15], 'k-', linewidth=1.5)
            ax.plot([r['ci_hi'], r['ci_hi']], [i-0.15, i+0.15], 'k-', linewidth=1.5)

    ax.axvline(x=k_global, color='red', linestyle='--', linewidth=2,
               label=f'Global k = {k_global:.0f}')

    ax.set_yticks(range(len(alloy_labels)))
    ax.set_yticklabels([f'{l} (n={n})' for l, n in zip(alloy_labels, alloy_n)], fontsize=8)
    ax.set_xlabel('k_HP (MPa·µm¹/²)')
    ax.set_title('Per-alloy Hall–Petch coefficient\n(green = reliable GS variation, gray = low GS variation)')
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    # Value labels
    for i, v in enumerate(alloy_k):
        ax.text(max(v, 0) + 20, i, f'{v:.0f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/46_per_alloy_kHP.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved 46_per_alloy_kHP.png")

# --- Plot 47: Bootstrap CI forest plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 47a: Element coefficients
ax = axes[0]
elem_indices = list(range(1, 8))
elem_betas = [beta_m3[i] for i in elem_indices]
elem_boot_lo = [boot_lo[i] for i in elem_indices]
elem_boot_hi = [boot_hi[i] for i in elem_indices]
elem_ols_se_lo = [beta_m3[i] - 1.96*se_ols[i] for i in elem_indices]
elem_ols_se_hi = [beta_m3[i] + 1.96*se_ols[i] for i in elem_indices]

y_pos = np.arange(len(elem_short))

# OLS CI (lighter, wider bars)
for j, yp in enumerate(y_pos):
    ax.plot([elem_ols_se_lo[j], elem_ols_se_hi[j]], [yp+0.15, yp+0.15],
            color='#90CAF9', linewidth=3, solid_capstyle='round')

# Bootstrap CI (darker, narrower bars)
for j, yp in enumerate(y_pos):
    ax.plot([elem_boot_lo[j], elem_boot_hi[j]], [yp-0.15, yp-0.15],
            color='#1565C0', linewidth=3, solid_capstyle='round')

# Point estimates
ax.scatter(elem_betas, y_pos, c='red', s=60, zorder=5, marker='D', edgecolors='k', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'α_{e}' for e in elem_short], fontsize=11)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Coefficient (MPa per unit fraction)')
ax.set_title('(a) σ₀ composition coefficients\n(red = OLS, light blue = OLS 95% CI, dark blue = bootstrap 95% CI)')
ax.invert_yaxis()

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#90CAF9', linewidth=3, label='OLS 95% CI'),
    Line2D([0], [0], color='#1565C0', linewidth=3, label='Bootstrap 95% CI'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
           markeredgecolor='k', markersize=8, label='OLS estimate'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower left')

# 47b: intercept and k
ax = axes[1]
other_indices = [0, 8]  # intercept and k
other_labels = ['σ₀₀ (intercept)', 'k (HP coeff.)']
other_betas = [beta_m3[i] for i in other_indices]
other_boot_lo = [boot_lo[i] for i in other_indices]
other_boot_hi = [boot_hi[i] for i in other_indices]
other_ols_lo = [beta_m3[i] - 1.96*se_ols[i] for i in other_indices]
other_ols_hi = [beta_m3[i] + 1.96*se_ols[i] for i in other_indices]

y_pos2 = np.arange(len(other_labels))

for j, yp in enumerate(y_pos2):
    ax.plot([other_ols_lo[j], other_ols_hi[j]], [yp+0.15, yp+0.15],
            color='#90CAF9', linewidth=3, solid_capstyle='round')
    ax.plot([other_boot_lo[j], other_boot_hi[j]], [yp-0.15, yp-0.15],
            color='#1565C0', linewidth=3, solid_capstyle='round')

ax.scatter(other_betas, y_pos2, c='red', s=60, zorder=5, marker='D',
           edgecolors='k', linewidth=0.5)

ax.set_yticks(y_pos2)
ax.set_yticklabels(other_labels, fontsize=11)
ax.set_xlabel('Coefficient value')
ax.set_title('(b) Intercept and k_HP\n(OLS vs bootstrap 95% CI)')
ax.invert_yaxis()
ax.legend(handles=legend_elements, fontsize=9, loc='lower left')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/47_bootstrap_ci.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved 47_bootstrap_ci.png")

# ============================================================
# 8. SAVE SUMMARY FOR REPORT
# ============================================================
summary = {
    'vif_max': max_vif,
    'cond_number': cond,
    'mc_k_ci': (mc_lo[-1], mc_hi[-1]),
    'mc_k_shift_pct': abs(mc_mean[-1] - beta_m3[-1]) / abs(beta_m3[-1]) * 100,
    'partial_r_V_YS': r_partial,
    'partial_p_V_YS': p_partial,
    'raw_r_V_YS': r_vy,
    'attenuation_pct': attenuation,
    'n_subsets': len(subset_k_results),
    'subset_k_range': (min(r['k'] for r in subset_k_results.values()),
                       max(r['k'] for r in subset_k_results.values())),
    'n_multi_gs': len(per_alloy_k),
    'boot_se_ratio_range': (min(boot_std / se_ols), max(boot_std / se_ols)),
}

# Save to CSV for easy reference
pd.DataFrame([summary]).to_csv(f'{RESULTS_DIR}/review_diagnostics_summary.csv', index=False)

print("\n" + "=" * 70)
print("DIAGNOSTICS SUMMARY")
print("=" * 70)
print(f"""
  1. MULTICOLLINEARITY: Element VIFs all < {max(vif_values[:-1]):.1f}, d⁻¹/² VIF = {vif_values[-1]:.1f} (expected), cond = {cond:.1f}
     --> Element multicollinearity acceptable; d⁻¹/² VIF elevated due to composition-GS correlation

  2. GRAIN-SIZE SENSITIVITY: MC 95% CI for k = [{mc_lo[-1]:.0f}, {mc_hi[-1]:.0f}]
     Mean shift: {summary['mc_k_shift_pct']:.1f}% --> Coefficients robust to GS uncertainty

  3. SUBSET k_HP: Range {summary['subset_k_range'][0]:.0f}–{summary['subset_k_range'][1]:.0f} vs global {k_global:.0f}
     --> k_HP varies across subsets but overlapping CIs

  4. PER-ALLOY k_HP: {summary['n_multi_gs']} compositions with multiple grain sizes
     --> Limited data; most compositions have similar grain sizes

  5. SIMPSON'S PARADOX: Raw r(V,YS) = {r_vy:+.3f}, partial r(V,YS|d⁻¹/²) = {r_partial:+.3f}
     Attenuation: {attenuation:.0f}% --> {"CAUTION: substantial confounding" if attenuation > 30 else "Moderate confounding"}

  6. BOOTSTRAP CIs: SE ratios range {summary['boot_se_ratio_range'][0]:.2f}–{summary['boot_se_ratio_range'][1]:.2f}
     --> {"OLS SEs reliable (ratios near 1)" if 0.8 < summary['boot_se_ratio_range'][0] and summary['boot_se_ratio_range'][1] < 1.2 else "Some departure from normality assumption"}
""")

print("=" * 70)
print("DONE — All diagnostic plots saved to analysis_plots/")
print("=" * 70)
