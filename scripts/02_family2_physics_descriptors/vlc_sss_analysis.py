#!/usr/bin/env python3
"""
Varvenne-Leyson-Curtin Solid Solution Strengthening Model
=========================================================
Physics-based computation of SSS for each HEA alloy.

Reference:
  Varvenne, Leyson, Curtin (2016) Acta Mater. 118, 164-176
  Varvenne, Curtin (2017) Scripta Mater. 142, 92-95

Model:
  τ_y0 = 0.051 α^(-1/3) μ (1+ν)/(1-ν))^(4/3) ·
         [Σ_n c_n ΔV_n²]^(2/3) / (b² Γ_bar^(1/3))

  Δσ_SSS = M · τ_y0 · [1 - (kT / (Δε_b · b³))^(2/3)]

  Simplified (T=0 limit for comparison):
  Δσ_SSS,0 = M · τ_y0

  Where:
  - α ~ 0.123 (line tension parameter)
  - μ = average shear modulus
  - ν = average Poisson's ratio
  - ΔV_n = atomic volume misfit of element n
  - b = Burgers vector = a/√2
  - Γ_bar = line tension ~ α μ b²
  - M = 3.06 (Taylor factor for FCC polycrystal)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# ============================================================
# ELEMENT PHYSICAL PROPERTIES
# ============================================================
# Metallic radii (pm) — Goldschmidt
RADII = {'Al': 143, 'Co': 125, 'Cr': 128, 'Cu': 128, 'Fe': 126, 'Mn': 127, 'Ni': 124, 'V': 134}

# Atomic volumes (Å³) — from FCC lattice parameters or experimental
# V = a³/4 for FCC
A_FCC = {'Al': 4.050, 'Co': 3.545, 'Cr': 3.520, 'Cu': 3.615, 'Fe': 3.590, 'Mn': 3.540, 'Ni': 3.524, 'V': 3.720}  # Fe: 3.590 Å (gamma-Fe FCC)
ATOMIC_VOL = {el: (a * 1e-10)**3 / 4 for el, a in A_FCC.items()}  # m³

# Shear modulus (GPa)
SHEAR_MOD = {'Al': 26, 'Co': 75, 'Cr': 115, 'Cu': 48, 'Fe': 82, 'Mn': 79, 'Ni': 76, 'V': 47}  # Co: 75 GPa (polycrystalline FCC)

# Bulk modulus (GPa) — for computing Poisson's ratio
BULK_MOD = {'Al': 76, 'Co': 180, 'Cr': 160, 'Cu': 140, 'Fe': 170, 'Mn': 120, 'Ni': 180, 'V': 158}

# Poisson's ratio = (3K - 2G) / (2(3K + G))
POISSON = {el: (3*BULK_MOD[el] - 2*SHEAR_MOD[el]) / (2*(3*BULK_MOD[el] + SHEAR_MOD[el]))
           for el in ELEMENTS}

# VEC
VEC_VALS = {'Al': 3, 'Co': 9, 'Cr': 6, 'Cu': 11, 'Fe': 8, 'Mn': 7, 'Ni': 10, 'V': 5}

# ============================================================
# VLC MODEL IMPLEMENTATION
# ============================================================

def compute_vlc_sss(row, T=300):
    """
    Compute Varvenne-Leyson-Curtin solid solution strengthening.

    Uses dimensionless misfit strains (δV_n = ΔV_n / V̄) per
    Varvenne & Curtin, Scripta Mater. 142 (2017) 92-95.

    Returns: dict with all intermediate quantities and final Δσ_SSS
    """
    fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
    active = {el: c for el, c in fracs.items() if c > 0}

    # Rule-of-mixtures average properties
    mu_bar_Pa = sum(c * SHEAR_MOD[el] * 1e9 for el, c in fracs.items())  # Pa
    mu_bar_GPa = mu_bar_Pa / 1e9
    nu_bar = sum(c * POISSON[el] for el, c in fracs.items())
    a_bar_m = sum(c * A_FCC[el] for el, c in fracs.items()) * 1e-10  # m
    b_m = a_bar_m / np.sqrt(2)  # Burgers vector (m)

    # Average atomic volume and metallic radius
    V_bar = sum(c * ATOMIC_VOL[el] for el, c in fracs.items())  # m³
    r_bar_pm = sum(c * RADII[el] for el, c in fracs.items())  # pm

    # --- Method 1: VLC with DIMENSIONLESS fractional volume misfits ---
    # δV_n = ΔV_n / V̄  (dimensionless)
    delta_V_frac = {el: (ATOMIC_VOL[el] - V_bar) / V_bar for el in ELEMENTS}

    # Concentration-weighted squared fractional misfit
    sigma_delta_sq = sum(c * delta_V_frac[el]**2 for el, c in fracs.items())

    # VLC parameters
    alpha = 0.123  # Line tension parameter
    M_taylor = 3.06  # Taylor factor for FCC polycrystal
    kB = 1.381e-23  # Boltzmann constant (J/K)

    # VLC zero-temperature CRSS (Varvenne & Curtin 2017, Eq. 3-4):
    # τ_y0 = 0.051 · α^(-1/3) · f₁^(2/3) · μ̄ · [Σ c_n δV_n²]^(2/3)
    # f₁(ν) = [(1+ν)/(1-ν)]² · 4/9  (pressure coupling for edge dislocations)
    f1 = ((1 + nu_bar) / (1 - nu_bar))**2 * 4.0 / 9.0

    tau_y0 = 0.051 * alpha**(-1.0/3.0) * f1**(2.0/3.0) * mu_bar_Pa * sigma_delta_sq**(2.0/3.0)

    # Polycrystalline yield stress (T=0)
    sigma_y0_VLC = M_taylor * tau_y0 / 1e6  # MPa

    # Finite-temperature correction
    # ΔE_b/V = 0.274 · Γ^(1/3) · [μ̄·b⁵·f₁·Σc_nδV_n²]^(2/3)
    # For practical VLC: σ(T) = σ(0) · [1 - (kT/ΔE_b)^(2/3)]
    Gamma = alpha * mu_bar_Pa * b_m**2  # line tension (N)
    Delta_Eb = 0.274 * Gamma**(1.0/3.0) * (mu_bar_Pa * b_m**5 * f1 * sigma_delta_sq)**(2.0/3.0)

    if Delta_Eb > 0 and T > 0:
        ratio = kB * T / Delta_Eb
        if ratio < 1.0:
            T_correction = (1 - ratio**(2.0/3.0))
        else:
            T_correction = 0.0  # athermal limit exceeded
    else:
        T_correction = 1.0

    sigma_y_VLC_T = sigma_y0_VLC * T_correction

    # --- Method 2: Radius-based misfit strains (Labusch-type) ---
    delta_r = {el: (RADII[el] - r_bar_pm) / r_bar_pm for el in ELEMENTS}
    delta_mu_i = {el: (SHEAR_MOD[el] - mu_bar_GPa) / mu_bar_GPa for el in ELEMENTS}

    # Combined Labusch parameter: ε² = δ_G² + α²·δ_r²
    # α_L ≈ 16 for edge dislocations in FCC
    alpha_L = 16
    eps_sq_sum = sum(c * (delta_mu_i[el]**2 + alpha_L**2 * delta_r[el]**2) for el, c in fracs.items())
    eps_Labusch = np.sqrt(eps_sq_sum)

    # Labusch SSS estimate: Δσ ≈ (2/3)^(2/3) · μ̄ · ε_L^(4/3) · c_eff^(1/3)
    # Using empirical scaling that gives ~MPa-range values
    n_comp = len(active)
    c_eff = 1.0 / n_comp  # effective solute concentration
    sigma_Labusch = mu_bar_GPa * 1000 * eps_Labusch**(4.0/3.0) * c_eff**(1.0/3.0)  # MPa

    # --- Method 3: Toda-Caraballo model ---
    # Δσ_TC = M · μ̄ · δ^(4/3) · c_eff^(1/3)
    # where δ = sqrt(Σ c_n · ((r_n - r̄)/r̄)²)
    delta_Yang = np.sqrt(sum(c * delta_r[el]**2 for el, c in fracs.items()))
    sigma_TC = M_taylor * mu_bar_GPa * 1000 * delta_Yang**(4.0/3.0) * c_eff**(1.0/3.0)  # MPa

    return pd.Series({
        'mu_bar_GPa': mu_bar_GPa,
        'nu_bar': nu_bar,
        'a_bar_A': a_bar_m * 1e10,
        'b_A': b_m * 1e10,
        'V_bar_A3': V_bar * 1e30,
        'sigma_delta_sq': sigma_delta_sq,
        'tau_y0_MPa': tau_y0 / 1e6,
        'sigma_y0_VLC': sigma_y0_VLC,  # T=0 K
        'sigma_y_VLC_300K': sigma_y_VLC_T,  # T=300 K
        'T_correction': T_correction,
        'Delta_Eb_eV': Delta_Eb / 1.602e-19,
        'eps_Labusch': eps_Labusch,
        'sigma_Labusch': sigma_Labusch,
        'sigma_TC': sigma_TC,
        'delta_Yang': delta_Yang,
    })


# ============================================================
# COMPUTE VLC SSS FOR ALL ALLOYS
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING VLC SSS FOR ALL ALLOYS")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
vlc = df.apply(compute_vlc_sss, axis=1)
df = pd.concat([df, vlc], axis=1)

print(f"\nVLC SSS Statistics:")
print(f"  σ_y0 (VLC, T=0):  {df['sigma_y0_VLC'].min():.1f} - {df['sigma_y0_VLC'].max():.1f} MPa "
      f"(mean={df['sigma_y0_VLC'].mean():.1f})")
print(f"  σ_y (VLC, 300K):  {df['sigma_y0_VLC'].min():.1f} - {df['sigma_y0_VLC'].max():.1f} MPa "
      f"(mean={df['sigma_y0_VLC'].mean():.1f})")
print(f"  σ_Labusch:         {df['sigma_Labusch'].min():.1f} - {df['sigma_Labusch'].max():.1f} MPa "
      f"(mean={df['sigma_Labusch'].mean():.1f})")
print(f"  σ_TC:              {df['sigma_TC'].min():.1f} - {df['sigma_TC'].max():.1f} MPa "
      f"(mean={df['sigma_TC'].mean():.1f})")
print(f"  ΔE_b (eV):         {df['Delta_Eb_eV'].min():.4f} - {df['Delta_Eb_eV'].max():.4f}")

# Per-element misfit contribution
print("\n  Per-element fractional volume misfits (δV_n = ΔV_n/V̄):")
for i, row in df.head(5).iterrows():
    alloy = row['Alloy']
    V_bar = sum(row[f'{el}_frac'] * ATOMIC_VOL[el] for el in ELEMENTS)
    misfits = {el: (ATOMIC_VOL[el] - V_bar) / V_bar for el in ELEMENTS if row[f'{el}_frac'] > 0}
    print(f"    {alloy}: " + ", ".join(f"{el}={dv:+.4f}" for el, dv in misfits.items()))

# ============================================================
# SUPERPOSITION MODELS: σ_y = σ_SSS + k_HP · d^(-1/2)
# ============================================================
print("\n" + "=" * 70)
print("SUPERPOSITION MODELS: σ_y = σ_SSS + k_HP · d^(-1/2)")
print("=" * 70)

df_ys = df.dropna(subset=['YS'])

# Model 1: σ_y = A·σ_VLC + k_HP·d^(-1/2)  (linear superposition)
# Model 2: σ_y = A·σ_VLC + k_HP·d^(-1/2) + C  (with offset)
# Model 3: σ_y = sqrt(σ_VLC² + (k_HP·d^(-1/2))²)  (RSS superposition)
# Model 4: σ_y = f(σ_VLC, d^(-1/2), compositions, processing)

# Linear superposition
from sklearn.linear_model import LinearRegression
X_sup1 = df_ys[['sigma_y0_VLC', 'd_inv_sqrt']].values
y_ys = df_ys['YS'].values
reg_sup1 = LinearRegression().fit(X_sup1, y_ys)
r2_sup1 = reg_sup1.score(X_sup1, y_ys)
print(f"\n  Model 1: YS = {reg_sup1.coef_[0]:.2f}·σ_VLC + {reg_sup1.coef_[1]:.1f}·d^(-1/2) "
      f"+ {reg_sup1.intercept_:.1f}")
print(f"  R² (in-sample): {r2_sup1:.4f}")

# With processing
X_sup2 = df_ys[['sigma_y0_VLC', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime']].values
reg_sup2 = LinearRegression().fit(X_sup2, y_ys)
r2_sup2 = reg_sup2.score(X_sup2, y_ys)
print(f"\n  Model 2: YS = σ_VLC + k_HP·d^(-1/2) + processing")
print(f"  Coefficients: {dict(zip(['σ_VLC', 'd^(-1/2)', 'CW', 'RecrT', 'HT'], reg_sup2.coef_))}")
print(f"  R² (in-sample): {r2_sup2:.4f}")

# With compositions + VLC + HP + processing
X_sup3 = df_ys[ELEMENTS + ['sigma_y0_VLC', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime']].values
reg_sup3 = LinearRegression().fit(X_sup3, y_ys)
r2_sup3 = reg_sup3.score(X_sup3, y_ys)
print(f"\n  Model 3: YS = compositions + σ_VLC + k_HP·d^(-1/2) + processing")
print(f"  R² (in-sample): {r2_sup3:.4f}")

# ============================================================
# LOO CROSS-VALIDATION OF ALL MODELS
# ============================================================
print("\n" + "=" * 70)
print("LOO CROSS-VALIDATION OF SUPERPOSITION MODELS")
print("=" * 70)

models_to_test = {
    'Hall-Petch only': ['d_inv_sqrt'],
    'VLC only': ['sigma_y0_VLC'],
    'Labusch only': ['sigma_Labusch'],
    'TC only': ['sigma_TC'],
    'VLC + HP': ['sigma_y0_VLC', 'd_inv_sqrt'],
    'Labusch + HP': ['sigma_Labusch', 'd_inv_sqrt'],
    'VLC + HP + Process': ['sigma_y0_VLC', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Comp + HP + Process': ELEMENTS + ['d_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Comp + VLC + HP + Proc': ELEMENTS + ['sigma_y0_VLC', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Comp + Labusch + HP + Proc': ELEMENTS + ['sigma_Labusch', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Comp + TC + HP + Proc': ELEMENTS + ['sigma_TC', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Comp + VLC + Labusch + HP + Proc': ELEMENTS + ['sigma_y0_VLC', 'sigma_Labusch',
                                                      'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Full physics': ELEMENTS + ['sigma_y0_VLC', 'sigma_Labusch', 'sigma_TC',
                                 'delta', 'VEC', 'dH_mix', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
}

print(f"\n  {'Model':<35s} {'LOO R²':>8s} {'LOO RMSE':>10s} {'LOO MAE':>10s}")
print(f"  {'-' * 65}")

results_table = []

for mname, features in models_to_test.items():
    df_clean = df_ys[features + ['YS']].replace([np.inf, -np.inf], np.nan).dropna()
    X = df_clean[features].values
    y = df_clean['YS'].values
    n = len(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge regression (stable for small n)
    loo_preds = np.zeros(n)
    for train_idx, test_idx in LeaveOneOut().split(X_scaled):
        lr = Ridge(alpha=1.0).fit(X_scaled[train_idx], y[train_idx])
        loo_preds[test_idx] = lr.predict(X_scaled[test_idx])

    r2_loo = r2_score(y, loo_preds)
    rmse_loo = np.sqrt(mean_squared_error(y, loo_preds))
    mae_loo = np.mean(np.abs(y - loo_preds))

    print(f"  {mname:<35s} {r2_loo:>8.4f} {rmse_loo:>10.2f} {mae_loo:>10.2f}")
    results_table.append({'model': mname, 'R2': r2_loo, 'RMSE': rmse_loo, 'MAE': mae_loo})

# Also test with Gradient Boosting
print(f"\n  {'--- Gradient Boosting ---':<35s}")
for mname, features in models_to_test.items():
    df_clean = df_ys[features + ['YS']].replace([np.inf, -np.inf], np.nan).dropna()
    X = df_clean[features].values
    y = df_clean['YS'].values
    n = len(y)

    loo_preds = np.zeros(n)
    for train_idx, test_idx in LeaveOneOut().split(X):
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        min_samples_leaf=5, random_state=42)
        gb.fit(X[train_idx], y[train_idx])
        loo_preds[test_idx] = gb.predict(X[test_idx])

    r2_loo = r2_score(y, loo_preds)
    rmse_loo = np.sqrt(mean_squared_error(y, loo_preds))
    mae_loo = np.mean(np.abs(y - loo_preds))

    print(f"  {mname:<35s} {r2_loo:>8.4f} {rmse_loo:>10.2f} {mae_loo:>10.2f}")

# ============================================================
# VISUALIZATION: VLC SSS vs Experimental
# ============================================================
print("\n" + "=" * 70)
print("GENERATING VLC PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
batch_colors = {'BBA': '#D55E00', 'BBB': '#0072B2', 'BBC': '#009E73',
                'CBA': '#CC79A7', 'CBB': '#E69F00', 'CBC': '#56B4E9'}

# (a) VLC SSS vs Experimental YS
ax = axes[0, 0]
for batch, color in batch_colors.items():
    mask = (df_ys['Iteration'] == batch)
    ax.scatter(df_ys.loc[mask, 'sigma_y0_VLC'], df_ys.loc[mask, 'YS'],
               c=color, label=batch, s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
r_val, p_val = stats.pearsonr(df_ys['sigma_y0_VLC'], df_ys['YS'])
ax.set_xlabel('VLC σ_SSS (300K) [MPa]', fontsize=12)
ax.set_ylabel('Experimental YS [MPa]', fontsize=12)
ax.set_title(f'VLC SSS vs Experimental YS (r={r_val:.3f})', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Residual YS after removing HP contribution vs VLC SSS
ax = axes[0, 1]
# Residual = YS - k_HP·d^(-1/2)
reg_hp = LinearRegression().fit(df_ys[['d_inv_sqrt']].values, y_ys)
ys_resid = y_ys - reg_hp.predict(df_ys[['d_inv_sqrt']].values)
for batch, color in batch_colors.items():
    mask = (df_ys['Iteration'] == batch)
    ax.scatter(df_ys.loc[mask, 'sigma_y0_VLC'], ys_resid[mask.values],
               c=color, label=batch, s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
r_resid, p_resid = stats.pearsonr(df_ys['sigma_y0_VLC'], ys_resid)
ax.set_xlabel('VLC σ_SSS (300K) [MPa]', fontsize=12)
ax.set_ylabel('YS − k_HP·d^(-1/2) [MPa]', fontsize=12)
ax.set_title(f'Residual YS (after HP) vs VLC SSS (r={r_resid:.3f})', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Parity plot: best superposition model
ax = axes[1, 0]
best_feats = ELEMENTS + ['sigma_y0_VLC', 'd_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime']
df_clean = df_ys[best_feats + ['YS']].dropna()
X_best = df_clean[best_feats].values
y_best = df_clean['YS'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_best)

loo_preds = np.zeros(len(y_best))
for train_idx, test_idx in LeaveOneOut().split(X_scaled):
    en = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000).fit(X_scaled[train_idx], y_best[train_idx])
    loo_preds[test_idx] = en.predict(X_scaled[test_idx])

r2_best = r2_score(y_best, loo_preds)
rmse_best = np.sqrt(mean_squared_error(y_best, loo_preds))

for batch, color in batch_colors.items():
    mask = (df_clean.merge(df_ys[['Alloy', 'Iteration']], left_index=True, right_index=True)['Iteration'] == batch).values
    if mask.sum() > 0:
        ax.scatter(y_best[mask], loo_preds[mask], c=color, label=batch, s=40, alpha=0.8,
                   edgecolors='k', linewidth=0.5)
lims = [min(y_best.min(), loo_preds.min()) * 0.9, max(y_best.max(), loo_preds.max()) * 1.1]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Experimental YS [MPa]', fontsize=12)
ax.set_ylabel('Predicted YS (LOO) [MPa]', fontsize=12)
ax.set_title(f'Comp + VLC + HP + Proc: R²={r2_best:.3f}, RMSE={rmse_best:.1f}', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# (d) VLC SSS colored by V content
ax = axes[1, 1]
sc = ax.scatter(df_ys['sigma_y0_VLC'], df_ys['YS'],
                c=df_ys['V'], cmap='plasma', s=60, alpha=0.8,
                edgecolors='k', linewidth=0.5)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('V content (at%)', fontsize=11)
ax.set_xlabel('VLC σ_SSS (300K) [MPa]', fontsize=12)
ax.set_ylabel('Experimental YS [MPa]', fontsize=12)
ax.set_title('VLC SSS vs YS (colored by V)', fontsize=13)
ax.grid(True, alpha=0.3)

plt.suptitle('VLC Solid Solution Strengthening Analysis', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/13_vlc_sss_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved VLC SSS analysis plots")

# ============================================================
# DECOMPOSED STRENGTHENING: σ_y = σ_0 + Δσ_SSS + Δσ_HP
# ============================================================
print("\n" + "=" * 70)
print("DECOMPOSED STRENGTHENING CONTRIBUTIONS")
print("=" * 70)

# Fit: YS = c0 + c1·σ_VLC + c2·d^(-1/2)
from sklearn.linear_model import LinearRegression
X_decomp = df_ys[['sigma_y0_VLC', 'd_inv_sqrt']].values
reg_decomp = LinearRegression().fit(X_decomp, y_ys)

sigma_0_base = reg_decomp.intercept_
c_vlc = reg_decomp.coef_[0]
k_hp_adj = reg_decomp.coef_[1]

df_ys_contrib = df_ys.copy()
df_ys_contrib['sigma_0'] = sigma_0_base
df_ys_contrib['delta_sigma_SSS'] = c_vlc * df_ys['sigma_y0_VLC']
df_ys_contrib['delta_sigma_HP'] = k_hp_adj * df_ys['d_inv_sqrt']
df_ys_contrib['YS_predicted'] = sigma_0_base + c_vlc * df_ys['sigma_y0_VLC'] + k_hp_adj * df_ys['d_inv_sqrt']

print(f"\n  σ_y = {sigma_0_base:.1f} + {c_vlc:.2f}·σ_VLC + {k_hp_adj:.1f}·d^(-1/2)")
print(f"\n  Strengthening decomposition (mean ± std):")
print(f"    σ_0 (base):     {sigma_0_base:.1f} MPa")
print(f"    Δσ_SSS (VLC):   {df_ys_contrib['delta_sigma_SSS'].mean():.1f} ± "
      f"{df_ys_contrib['delta_sigma_SSS'].std():.1f} MPa")
print(f"    Δσ_HP (grain):  {df_ys_contrib['delta_sigma_HP'].mean():.1f} ± "
      f"{df_ys_contrib['delta_sigma_HP'].std():.1f} MPa")
print(f"    Total predicted: {df_ys_contrib['YS_predicted'].mean():.1f} ± "
      f"{df_ys_contrib['YS_predicted'].std():.1f} MPa")
print(f"    Total exper.:    {df_ys_contrib['YS'].mean():.1f} ± {df_ys_contrib['YS'].std():.1f} MPa")

# Stacked bar chart
fig, ax = plt.subplots(figsize=(16, 6))
idx = np.arange(len(df_ys_contrib))
alloys = df_ys_contrib['Alloy'].values

# Sort by experimental YS
sort_idx = np.argsort(df_ys_contrib['YS'].values)
ax.bar(idx, [sigma_0_base]*len(idx), label=f'σ₀ = {sigma_0_base:.0f} MPa', color='#B0BEC5')
ax.bar(idx, df_ys_contrib['delta_sigma_SSS'].values[sort_idx], bottom=sigma_0_base,
       label=f'Δσ_SSS (VLC)', color='#FF7043')
ax.bar(idx, df_ys_contrib['delta_sigma_HP'].values[sort_idx],
       bottom=sigma_0_base + df_ys_contrib['delta_sigma_SSS'].values[sort_idx],
       label=f'Δσ_HP (Hall-Petch)', color='#42A5F5')
ax.scatter(idx, df_ys_contrib['YS'].values[sort_idx], c='k', s=20, zorder=5, label='Exp. YS')

ax.set_xlabel('Alloy (sorted by YS)', fontsize=12)
ax.set_ylabel('Yield Strength (MPa)', fontsize=12)
ax.set_title('Decomposed Strengthening: σ₀ + Δσ_SSS + Δσ_HP', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(idx[::5])
ax.set_xticklabels(alloys[sort_idx][::5], rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/14_strengthening_decomposition.png', dpi=150)
plt.close()
print("  Saved strengthening decomposition plot")

# Save augmented data
df.to_csv(f'{DATA_DIR}/data_with_vlc.csv', index=False)
print(f"\n  Saved augmented data with VLC to data_with_vlc.csv")

print("\n" + "=" * 70)
print("VLC SSS ANALYSIS COMPLETE")
print("=" * 70)
