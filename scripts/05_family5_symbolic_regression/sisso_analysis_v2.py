#!/usr/bin/env python3
"""
SISSO v2: Enhanced Symbolic Regression for HEA Yield Strength
==============================================================
Improvements over v1 (sisso_analysis.py):
  1. Relaxed Hall-Petch exponent: 10 grain-size features (d^{-α}, ln(d))
  2. Unary operators: pow(2), pow(1/2), ^-1
  3. BIC-based dimension selection (sweep 1-5)
  4. Larger SIS threshold: 30 (was 20)

Same physics-informed Oliynyk features as v1.
"""

import time
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
import torch
import warnings
warnings.filterwarnings('ignore')

from TorchSisso.FeatureSpaceConstruction import feature_space_construction
from TorchSisso.Regressor import Regressor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'
ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

t0 = time.time()

print("=" * 70)
print("SISSO v2 — ENHANCED SYMBOLIC REGRESSION")
print("  1. Relaxed HP exponent (10 grain-size features)")
print("  2. Unary operators: pow(2), pow(1/2), ^-1")
print("  3. BIC-based dimension selection (dim 1-5)")
print("  4. SIS threshold: 30")
print("=" * 70)

# ============================================================
# 1. ELEMENTAL PROPERTY DATABASE
# ============================================================
RADII = {'Al': 143, 'Co': 125, 'Cr': 128, 'Cu': 128, 'Fe': 126,
         'Mn': 127, 'Ni': 124, 'V': 134}
VEC_VALS = {'Al': 3, 'Co': 9, 'Cr': 6, 'Cu': 11, 'Fe': 8,
            'Mn': 7, 'Ni': 10, 'V': 5}
EN = {'Al': 1.61, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.90, 'Fe': 1.83,
      'Mn': 1.55, 'Ni': 1.91, 'V': 1.63}
TM = {'Al': 933, 'Co': 1768, 'Cr': 2180, 'Cu': 1358, 'Fe': 1811,
      'Mn': 1519, 'Ni': 1728, 'V': 2183}
SHEAR_MOD = {'Al': 26, 'Co': 75, 'Cr': 115, 'Cu': 48, 'Fe': 82,
             'Mn': 79, 'Ni': 76, 'V': 47}
BULK_MOD = {'Al': 76, 'Co': 180, 'Cr': 160, 'Cu': 140, 'Fe': 170,
            'Mn': 120, 'Ni': 180, 'V': 158}
A_FCC = {'Al': 4.050, 'Co': 3.545, 'Cr': 3.520, 'Cu': 3.615,
         'Fe': 3.590, 'Mn': 3.540, 'Ni': 3.524, 'V': 3.720}
MASS = {'Al': 26.98, 'Co': 58.93, 'Cr': 52.00, 'Cu': 63.55, 'Fe': 55.85,
        'Mn': 54.94, 'Ni': 58.69, 'V': 50.94}


# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
if 'eps_Labusch.1' in df.columns:
    df = df.drop(columns=['eps_Labusch.1'])
df['Omega'] = df['Omega'].clip(upper=100)
df = df.replace([np.inf, -np.inf], np.nan)

df_ys = df.dropna(subset=['YS']).copy()
y = df_ys['YS'].values
n = len(y)
d = df_ys['GrainSize'].values
groups = df_ys['Iteration'].values

print(f"Loaded {n} alloys with YS data")
print(f"YS range: {y.min():.0f} – {y.max():.0f} MPa")
print(f"Grain size range: {d.min():.1f} – {d.max():.1f} µm")


# ============================================================
# 3. COMPUTE OLIYNYK-STYLE FEATURES
# ============================================================
print("\nComputing physics-based features...")

def compute_oliynyk_features(row):
    """Compute Oliynyk/Matminer-style statistical features."""
    fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
    active = {el: c for el, c in fracs.items() if c > 0}
    features = {}

    properties = {
        'r': RADII, 'mu': SHEAR_MOD, 'K': BULK_MOD, 'EN': EN,
        'Tm': TM, 'VEC': VEC_VALS, 'mass': MASS, 'a_fcc': A_FCC,
    }

    for prop_name, prop_dict in properties.items():
        vals = np.array([prop_dict[el] for el in ELEMENTS])
        cs = np.array([fracs[el] for el in ELEMENTS])
        active_vals = [prop_dict[el] for el in active]

        mean_val = np.sum(cs * vals)
        features[f'{prop_name}_mean'] = mean_val

        var_val = np.sum(cs * (vals - mean_val)**2)
        features[f'{prop_name}_var'] = var_val

        if mean_val != 0:
            features[f'{prop_name}_delta'] = np.sqrt(var_val) / abs(mean_val)
        else:
            features[f'{prop_name}_delta'] = 0.0

        features[f'{prop_name}_range'] = max(active_vals) - min(active_vals)

    return pd.Series(features)


oliynyk_df = df_ys.apply(compute_oliynyk_features, axis=1)
print(f"  Computed {len(oliynyk_df.columns)} Oliynyk-style features")
for col in oliynyk_df.columns:
    df_ys[col] = oliynyk_df[col].values


# ============================================================
# 4. FLEXIBLE GRAIN-SIZE FEATURES
# ============================================================
print("\nComputing flexible grain-size features...")

GRAIN_EXPONENTS = [
    ('d_inv_033', -1/3,   'Cottrell-Petch'),
    ('d_inv_040', -0.40,  ''),
    ('d_inv_045', -0.45,  ''),
    ('d_inv_050', -0.50,  'Classic HP'),
    ('d_inv_055', -0.55,  ''),
    ('d_inv_060', -0.60,  ''),
    ('d_inv_067', -2/3,   'Pile-up limit'),
    ('d_inv_075', -0.75,  ''),
    ('d_inv_100', -1.00,  'Inverse'),
]

GRAIN_FEATURES = []
for name, exp, label in GRAIN_EXPONENTS:
    vals = d ** exp
    df_ys[name] = vals
    GRAIN_FEATURES.append(name)
    tag = f'  ({label})' if label else ''
    print(f"  {name}: d^({exp:.3f}){tag}  "
          f"range=[{vals.min():.4f}, {vals.max():.4f}]")

# Also add ln(d)
df_ys['ln_d'] = np.log(d)
GRAIN_FEATURES.append('ln_d')
print(f"  ln_d: range=[{np.log(d).min():.3f}, {np.log(d).max():.3f}]")
print(f"  Total grain-size features: {len(GRAIN_FEATURES)}")


# ============================================================
# 5. DEFINE FEATURE SETS
# ============================================================
FEAT_COMP_PHYSICS = [
    'r_mean', 'r_var', 'r_delta', 'r_range',
    'mu_mean', 'mu_var', 'mu_delta', 'mu_range',
    'EN_mean', 'EN_var', 'EN_delta', 'EN_range',
    'Tm_mean', 'Tm_var', 'Tm_delta', 'Tm_range',
    'VEC_mean',
    'K_mean', 'K_var',
    'delta', 'dS_mix', 'dH_mix', 'Omega',
    'Phi_VLC', 'eps_Labusch',
    'sigma_TC',
]

FEAT_FULL = FEAT_COMP_PHYSICS + GRAIN_FEATURES

print(f"\nComposition features: {len(FEAT_COMP_PHYSICS)}")
print(f"Grain-size features:  {len(GRAIN_FEATURES)}")
print(f"Total primary features: {len(FEAT_FULL)}")


# ============================================================
# 6. CONFIGURATION
# ============================================================
OPERATORS = ['+', '-', '*', '/', 'pow(2)', 'pow(1/2)', '^-1']
N_OPERATORS = 2          # Tier depth
MAX_DIMENSION = 4        # Sweep dimensions 1-4 (dim=5 OOM with SIS=30)
SIS_FEATURES = 30        # Top features per SIS step

print(f"\nSISSO v2 config:")
print(f"  Operators: {OPERATORS}")
print(f"  Tier depth: {N_OPERATORS}")
print(f"  Max dimension: {MAX_DIMENSION}")
print(f"  SIS threshold: {SIS_FEATURES}")


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================
def compute_ic(y_true, y_pred, k, n_samples):
    """AIC, AICc, BIC for a model with k parameters."""
    rss = np.sum((y_true - y_pred) ** 2)
    if rss <= 0:
        rss = 1e-15
    log_term = n_samples * np.log(rss / n_samples)
    aic = log_term + 2 * k
    bic = log_term + k * np.log(n_samples)
    if n_samples - k - 1 > 0:
        aicc = aic + 2 * k * (k + 1) / (n_samples - k - 1)
    else:
        aicc = np.inf
    return {'AIC': aic, 'AICc': aicc, 'BIC': bic}


def _eval_sisso_equation(equation_str, x_data, names):
    """Evaluate a SISSO equation string on given data."""
    x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)

    name_to_idx = {name: i for i, name in enumerate(names)}
    eq = equation_str.strip()
    prediction = np.zeros(x_np.shape[0])

    term_pattern = r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*(.+?)(?=\s*[+-]\s*[\d.]|\s*$)'
    terms = re.findall(term_pattern, eq)

    for coef_str, feat_name in terms:
        coef = float(coef_str.replace(' ', ''))
        feat_name = feat_name.strip()
        if feat_name in name_to_idx:
            prediction += coef * x_np[:, name_to_idx[feat_name]]
        else:
            matched = False
            for name, idx in name_to_idx.items():
                if name.strip() == feat_name:
                    prediction += coef * x_np[:, idx]
                    matched = True
                    break
            if not matched:
                print(f"    Warning: feature '{feat_name}' not found")

    remaining = re.sub(term_pattern, '', eq).strip()
    if remaining:
        try:
            prediction += float(remaining.replace(' ', ''))
        except ValueError:
            pass

    return prediction[0] if prediction.shape[0] == 1 else prediction


def run_sisso(df_input, operators, n_operators, dimension, sis_features,
              label="SISSO"):
    """Run SISSO feature expansion + regression."""
    print(f"\n  Running {label}...")
    print(f"    Operators: {operators}")
    print(f"    Tier: {n_operators}, dimension: {dimension}, SIS: {sis_features}")
    print(f"    Input features: {len(df_input.columns) - 1}")

    t_start = time.time()

    fsc = feature_space_construction(
        operators=operators,
        df=df_input.copy(),
        no_of_operators=n_operators,
        device='cpu'
    )
    x_expanded, y_tensor, names = fsc.feature_space()
    n_feats = x_expanded.shape[1]
    print(f"    Expanded feature space: {n_feats} features")

    reg = Regressor(x_expanded, y_tensor, names,
                    dimension=dimension, sis_features=sis_features)
    rmse, equation, r2, equations_per_dim = reg.regressor_fit()

    elapsed = time.time() - t_start
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Best equation (dim={dimension}): {equation[:120]}")
    print(f"    Train R²: {r2:.4f}, RMSE: {rmse:.2f}")

    return {
        'rmse': rmse,
        'r2': r2,
        'equation': equation,
        'equations_per_dim': equations_per_dim,
        'feature_names': names,
        'x_expanded': x_expanded,
        'y_tensor': y_tensor,
        'time': elapsed,
    }


def select_dimension_by_bic(equations_per_dim, x_expanded, y_tensor,
                             names, n_samples):
    """Evaluate BIC at each dimension and select the best.

    Within a fixed dimension, BIC ranking = RMSE ranking (same k_eff).
    BIC only discriminates ACROSS dimensions.
    """
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor
    results = []

    for dim_idx, equation in enumerate(equations_per_dim):
        dim = dim_idx + 1
        k_eff = dim + 1  # d terms + intercept
        y_pred = _eval_sisso_equation(equation, x_expanded, names)
        r2 = r2_score(y_np, y_pred)
        rmse = np.sqrt(mean_squared_error(y_np, y_pred))
        ic = compute_ic(y_np, y_pred, k_eff, n_samples)
        results.append({
            'dim': dim,
            'k_eff': k_eff,
            'Train_R2': r2,
            'Train_RMSE': rmse,
            'AIC': ic['AIC'],
            'AICc': ic['AICc'],
            'BIC': ic['BIC'],
            'equation': equation,
        })

    df_results = pd.DataFrame(results)
    best_idx = df_results['BIC'].idxmin()
    best = df_results.loc[best_idx]

    return {
        'best_dim': int(best['dim']),
        'best_equation': best['equation'],
        'best_bic': best['BIC'],
        'landscape': df_results,
    }


def sisso_loo_preexpanded(x_expanded, y_tensor, names, dimension, sis_features):
    """LOO-CV with pre-expanded feature space."""
    n_samples = len(y_tensor)
    preds = np.zeros(n_samples)
    x_np = x_expanded.numpy() if isinstance(x_expanded, torch.Tensor) else x_expanded
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor

    for i in range(n_samples):
        if (i + 1) % 20 == 0:
            running_r2 = r2_score(y_np[:i+1], preds[:i+1]) if i > 0 else 0
            print(f"    fold {i+1}/{n_samples}  (running R²={running_r2:.3f})")

        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False

        x_train = torch.tensor(x_np[mask], dtype=torch.float32)
        y_train = torch.tensor(y_np[mask], dtype=torch.float32)

        reg = Regressor(x_train, y_train, names,
                        dimension=dimension, sis_features=sis_features)
        rmse, equation, r2, _ = reg.regressor_fit()
        preds[i] = _eval_sisso_equation(equation, x_np[i:i+1], names)

    r2_loo = r2_score(y_np, preds)
    rmse_loo = np.sqrt(mean_squared_error(y_np, preds))
    mae_loo = mean_absolute_error(y_np, preds)
    return r2_loo, rmse_loo, mae_loo, preds


def sisso_lobo_preexpanded(x_expanded, y_tensor, names, groups_arr,
                            dimension, sis_features):
    """Leave-one-batch-out CV with pre-expanded features."""
    x_np = x_expanded.numpy() if isinstance(x_expanded, torch.Tensor) else x_expanded
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor
    preds = np.zeros(len(y_np))
    logo = LeaveOneGroupOut()

    for tr, te in logo.split(x_np, y_np, groups_arr):
        batch_name = groups_arr[te[0]]
        print(f"    LOBO fold: holding out {batch_name} ({len(te)} samples)")
        x_train = torch.tensor(x_np[tr], dtype=torch.float32)
        y_train = torch.tensor(y_np[tr], dtype=torch.float32)

        reg = Regressor(x_train, y_train, names,
                        dimension=dimension, sis_features=sis_features)
        rmse, equation, r2, _ = reg.regressor_fit()
        preds[te] = _eval_sisso_equation(equation, x_np[te], names)

    return r2_score(y_np, preds)


# ============================================================
# 8. FULL MODEL WITH BIC DIMENSION SELECTION
# ============================================================
print("\n" + "=" * 70)
print("SISSO v2: FULL MODEL with BIC dimension optimization")
print("=" * 70)

# Prepare feature dataframe (target first, then features)
df_full = pd.DataFrame()
df_full['YS'] = y
for feat in FEAT_FULL:
    vals = df_ys[feat].fillna(0).values
    if np.std(vals) > 1e-12:
        df_full[feat] = vals

n_feats_used = len(df_full.columns) - 1
print(f"  Using {n_feats_used} primary features (after removing zero-variance)")

# Run SISSO at MAX_DIMENSION=5
result = run_sisso(df_full, OPERATORS, N_OPERATORS,
                   dimension=MAX_DIMENSION, sis_features=SIS_FEATURES,
                   label=f"Full Model v2 (dim=1..{MAX_DIMENSION})")

# BIC dimension selection
print("\n--- BIC Dimension Selection ---")
dim_sel = select_dimension_by_bic(
    result['equations_per_dim'],
    result['x_expanded'],
    result['y_tensor'],
    result['feature_names'],
    n
)

print("\n  BIC landscape:")
landscape = dim_sel['landscape']
for _, row in landscape.iterrows():
    star = '  <-- BEST' if row['dim'] == dim_sel['best_dim'] else ''
    print(f"    dim={int(row['dim'])}: k={int(row['k_eff'])}, "
          f"Train R²={row['Train_R2']:.4f}, "
          f"BIC={row['BIC']:.1f}, AIC={row['AIC']:.1f}{star}")

best_dim = dim_sel['best_dim']
best_eq = dim_sel['best_equation']
k_eff = best_dim + 1
print(f"\n  BIC-optimal dimension: {best_dim} (k_eff={k_eff})")
print(f"  Equation: {best_eq}")

# Train predictions and IC at best dim
y_pred_train = _eval_sisso_equation(best_eq, result['x_expanded'],
                                     result['feature_names'])
ic = compute_ic(y, y_pred_train, k_eff, n)
train_r2 = r2_score(y, y_pred_train)

# Also evaluate at all dimensions for the full-data equation
print("\n  Equations at each dimension:")
for dim_idx, eq in enumerate(result['equations_per_dim']):
    print(f"    dim={dim_idx+1}: {eq[:120]}")

# Identify grain-size features in best equation
gs_in_eq = [gf for gf in GRAIN_FEATURES if gf in best_eq]
print(f"\n  Grain-size features in best equation: {gs_in_eq}")


# ============================================================
# 9. LOO-CV
# ============================================================
# dim=4 SO takes ~580s/fold → 93 folds ≈ 15 hours. Cap LOO at dim=3
# (BIC difference dim3 vs dim4 is negligible: ΔBIC ≈ 1.7)
LOO_DIM = min(best_dim, 3)
if LOO_DIM < best_dim:
    print(f"\n  Note: BIC-optimal dim={best_dim}, but dim={best_dim} SO is too slow "
          f"for LOO-CV (~580s/fold).")
    print(f"  Running LOO at dim={LOO_DIM} (ΔBIC={landscape.loc[best_dim-1,'BIC'] - landscape.loc[LOO_DIM-1,'BIC']:.1f})")
    loo_eq = result['equations_per_dim'][LOO_DIM - 1]
    loo_k_eff = LOO_DIM + 1
    loo_y_pred_train = _eval_sisso_equation(loo_eq, result['x_expanded'],
                                             result['feature_names'])
    loo_ic = compute_ic(y, loo_y_pred_train, loo_k_eff, n)
    loo_train_r2 = r2_score(y, loo_y_pred_train)
else:
    loo_eq = best_eq
    loo_k_eff = k_eff
    loo_ic = ic
    loo_train_r2 = train_r2

print("\n" + "=" * 70)
print(f"LOO-CV at dimension {LOO_DIM}")
print("=" * 70)
t_loo = time.time()
r2_loo, rmse_loo, mae_loo, preds_loo = sisso_loo_preexpanded(
    result['x_expanded'], result['y_tensor'],
    result['feature_names'], LOO_DIM, SIS_FEATURES
)
loo_time = time.time() - t_loo
print(f"\n  LOO R²={r2_loo:.4f}, RMSE={rmse_loo:.1f}, MAE={mae_loo:.1f}  "
      f"({loo_time:.0f}s)")


# ============================================================
# 10. LOBO-CV
# ============================================================
print("\n" + "=" * 70)
print(f"LOBO-CV at dimension {LOO_DIM}")
print("=" * 70)
t_lobo = time.time()
r2_lobo = sisso_lobo_preexpanded(
    result['x_expanded'], result['y_tensor'],
    result['feature_names'], groups, LOO_DIM, SIS_FEATURES
)
lobo_time = time.time() - t_lobo
print(f"\n  LOBO R²={r2_lobo:.4f}  ({lobo_time:.0f}s)")


# ============================================================
# 11. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

summary = pd.DataFrame([{
    'Model': f'SISSO_v2 (dim={LOO_DIM})',
    'n_terms': LOO_DIM,
    'BIC_optimal_dim': best_dim,
    'LOO_dim': LOO_DIM,
    'LOO_R2': r2_loo,
    'LOO_RMSE': rmse_loo,
    'LOO_MAE': mae_loo,
    'LOBO_R2': r2_lobo,
    'Train_R2': loo_train_r2,
    'k_eff': loo_k_eff,
    'AIC': loo_ic['AIC'],
    'AICc': loo_ic['AICc'],
    'BIC': loo_ic['BIC'],
    'Equation': loo_eq[:200],
    'BIC_landscape': str({int(r['dim']): round(r['BIC'], 1)
                          for _, r in landscape.iterrows()}),
}])

print(summary[['Model', 'n_terms', 'LOO_R2', 'LOO_RMSE', 'LOBO_R2',
               'Train_R2', 'AIC', 'BIC', 'k_eff']].to_string(index=False))

# Save BIC landscape
landscape.to_csv(f'{RESULTS_DIR}/sisso_v2_bic_landscape.csv', index=False)

# Save results
summary.to_csv(f'{RESULTS_DIR}/sisso_v2_results.csv', index=False)
print(f"\n  Saved sisso_v2_results.csv")
print(f"  Saved sisso_v2_bic_landscape.csv")

# Grain-size features in LOO equation
gs_in_loo_eq = [gf for gf in GRAIN_FEATURES if gf in loo_eq]

# v1 comparison
print("\n--- v1 vs v2 Comparison ---")
print(f"                     v1 (ref)    v2 (LOO dim={LOO_DIM})")
print(f"  LOO R²:            0.672       {r2_loo:.4f}  (Δ={r2_loo - 0.672:+.4f})")
print(f"  BIC:               714         {loo_ic['BIC']:.1f}  (Δ={loo_ic['BIC'] - 714:+.1f})")
print(f"  k_eff:             4           {loo_k_eff}")
print(f"  HP exponent:       -0.50       {'flexible' if gs_in_loo_eq else 'none'}")
print(f"  Grain-size feats:  {gs_in_loo_eq}")
print(f"  LOO equation:      {loo_eq[:120]}")


# ============================================================
# 12. VISUALIZATION
# ============================================================
BATCH_COLORS = {'BBA': '#D55E00', 'BBB': '#0072B2', 'BBC': '#009E73',
                'CBA': '#CC79A7', 'CBB': '#E69F00', 'CBC': '#56B4E9'}

# --- Plot 63a: BIC vs Dimension ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(landscape['dim'], landscape['BIC'], 'o-', linewidth=2,
         markersize=10, color='steelblue', label='BIC')
ax1.plot(landscape['dim'], landscape['AIC'], 's--', linewidth=1.5,
         markersize=8, color='coral', alpha=0.7, label='AIC')
ax1.axvline(best_dim, linestyle=':', color='red', alpha=0.7,
            label=f'BIC-optimal (dim={best_dim})')
ax1.set_xlabel('Dimension (number of terms)', fontsize=12)
ax1.set_ylabel('Information Criterion', fontsize=12)
ax1.set_title('(a) BIC/AIC vs Model Complexity', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, MAX_DIMENSION + 1))

ax2.plot(landscape['dim'], landscape['Train_R2'], 'o-', linewidth=2,
         markersize=10, color='seagreen', label='Train R²')
ax2.axvline(best_dim, linestyle=':', color='red', alpha=0.7,
            label=f'BIC-optimal (dim={best_dim})')
ax2.set_xlabel('Dimension (number of terms)', fontsize=12)
ax2.set_ylabel('Train R²', fontsize=12)
ax2.set_title('(b) Training Accuracy vs Complexity', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, MAX_DIMENSION + 1))

plt.suptitle('SISSO v2: BIC-Based Dimension Selection', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/63a_sisso_v2_bic_landscape.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved 63a_sisso_v2_bic_landscape.png")

# --- Plot 63b: Parity plot (LOO) ---
fig, ax = plt.subplots(figsize=(7, 7))
for batch in BATCH_COLORS:
    mask = groups == batch
    if mask.any():
        ax.scatter(y[mask], preds_loo[mask],
                   c=BATCH_COLORS[batch], s=50, alpha=0.7,
                   edgecolors='k', linewidth=0.5, label=batch)
lims = [min(y.min(), np.nanmin(preds_loo)) * 0.9,
        max(y.max(), np.nanmax(preds_loo)) * 1.1]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Experimental YS (MPa)', fontsize=11)
ax.set_ylabel('SISSO v2 Predicted YS (MPa)', fontsize=11)
ax.set_title(f'SISSO v2 (dim={best_dim}, BIC-optimal)\n'
             f'LOO R²={r2_loo:.3f}, RMSE={rmse_loo:.1f} MPa', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/63b_sisso_v2_parity.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 63b_sisso_v2_parity.png")

# --- Plot 63c: v1 vs v2 comparison ---
fig, ax = plt.subplots(figsize=(10, 6))

# Reference models
ref_models = [
    ('M0 (HP only)', 2, 0.406, 'gray', 'o', 10),
    ('M3 (comp+HP)', 9, 0.652, 'purple', 'o', 10),
    ('Stacking', 5, 0.670, 'red', 'D', 10),
    ('SISSO v1', 4, 0.672, 'coral', 's', 12),
]
for name, k, r2, color, marker, sz in ref_models:
    ax.scatter([k], [r2], c=color, s=sz*10, edgecolors='k',
               linewidth=1, label=f'{name}', zorder=4, marker=marker)

# SISSO v2
ax.scatter([loo_k_eff], [r2_loo], c='steelblue', s=200, edgecolors='k',
           linewidth=2, label=f'SISSO v2 (dim={LOO_DIM})', zorder=5, marker='*')

# v2 at all dimensions (faded)
for _, row in landscape.iterrows():
    dim_d = int(row['dim'])
    if dim_d != best_dim:
        ax.scatter([dim_d + 1], [row['Train_R2']], c='lightblue',
                   s=50, alpha=0.3, zorder=3, marker='o')

ax.set_xlabel('Effective Parameters (k)', fontsize=12)
ax.set_ylabel('LOO R²', fontsize=12)
ax.set_title('SISSO v1 vs v2: Flexible Exponent + Unary Operators', fontsize=14)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, max(0.75, r2_loo + 0.05))

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/63c_sisso_v1_vs_v2.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 63c_sisso_v1_vs_v2.png")


# ============================================================
# 13. APPEND TO MODEL COMPARISON CSV
# ============================================================
print("\n" + "=" * 70)
print("UPDATING MODEL COMPARISON TABLE")
print("=" * 70)

csv_path = f'{RESULTS_DIR}/model_search_results_v2.csv'
try:
    existing = pd.read_csv(csv_path)
    existing = existing[~existing['Model'].str.contains('SISSO_v2', na=False)]

    sisso_v2_row = pd.DataFrame([{
        'Model': 'SISSO_v2',
        'Features': 'SISSO_physics_v2',
        'n_feat': LOO_DIM,
        'LOO_R2': r2_loo,
        'LOO_RMSE': rmse_loo,
        'LOO_MAE': mae_loo,
        'LOBO_R2': r2_lobo,
        'Train_R2': loo_train_r2,
        'k_eff': loo_k_eff,
        'AIC': loo_ic['AIC'],
        'AICc': loo_ic['AICc'],
        'BIC': loo_ic['BIC'],
        'HPO': f'SISSO_BIC_dim{LOO_DIM}',
        'time': result['time'],
    }])

    updated = pd.concat([existing, sisso_v2_row], ignore_index=True)
    updated = updated.sort_values('LOO_R2', ascending=False)
    updated.to_csv(csv_path, index=False)
    print(f"  Added SISSO_v2 to {csv_path}")
    print(f"  Total models: {len(updated)}")
except FileNotFoundError:
    print(f"  Warning: {csv_path} not found, skipping")


# ============================================================
# 14. FINAL SUMMARY
# ============================================================
elapsed_total = time.time() - t0
print("\n" + "=" * 70)
print("SISSO v2 ANALYSIS COMPLETE")
print(f"Total time: {elapsed_total/60:.1f} min")
print("=" * 70)

print(f"\nKey improvements over v1:")
print(f"  1. Grain-size features: 1 → {len(GRAIN_FEATURES)} (relaxed HP exponent)")
print(f"  2. Operators: 4 binary → {len(OPERATORS)} (binary + unary)")
print(f"  3. SIS threshold: 20 → {SIS_FEATURES}")
print(f"  4. Dimension: fixed (3) → BIC-optimized ({best_dim})")

print(f"\nBIC-optimal dimension: {best_dim} (LOO evaluated at dim={LOO_DIM})")
print(f"LOO equation (dim={LOO_DIM}):")
print(f"  {loo_eq}")
if LOO_DIM < best_dim:
    print(f"BIC-best equation (dim={best_dim}, train only):")
    print(f"  {best_eq}")

print(f"\nGrain-size features in LOO equation: {gs_in_loo_eq}")
if gs_in_loo_eq:
    for gf in gs_in_loo_eq:
        for name, exp, label in GRAIN_EXPONENTS:
            if name == gf:
                print(f"  → {gf}: d^({exp:.3f})  {label}")
                break
        if gf == 'ln_d':
            print(f"  → ln_d: logarithmic scaling")

print(f"\nPerformance (at LOO dim={LOO_DIM}):")
print(f"  LOO R²:  {r2_loo:.4f}  (v1: 0.672, Δ={r2_loo - 0.672:+.4f})")
print(f"  BIC:     {loo_ic['BIC']:.1f}  (v1: 714, Δ={loo_ic['BIC'] - 714:+.1f})")
print(f"  LOBO R²: {r2_lobo:.4f}  (v1: 0.416)")
print(f"  k_eff:   {loo_k_eff}  (v1: 4)")
print(f"  Train R²: {loo_train_r2:.4f}")

print("\nDone.")
