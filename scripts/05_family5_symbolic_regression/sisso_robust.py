#!/usr/bin/env python3
"""
Robust SISSO Model Search
==========================

Re-runs SISSO with modified feature subsets to find equations that avoid
the EN_var/delta_mu singularity while maintaining predictive accuracy.

Variants tested:
  1. v1_baseline:  Original 30 features (reproduces existing result)
  2. no_delta_mu:  Remove delta_mu, mu_delta (28 features)
  3. no_deltas:    Also remove delta, r_delta, EN_delta, Tm_delta (24 features)
  4. swap_mu_var:  Replace delta_mu → mu_var, mu_delta → sqrt(mu_var) (30 features)
  5. v2_dim3:      Evaluate pre-computed v2 equation (no re-run)

Usage:
    python -u sisso_robust.py
"""

import time
import re
import sys
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut

from TorchSisso.FeatureSpaceConstruction import feature_space_construction
from TorchSisso.Regressor import Regressor

# Add project to path for external_validation imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
sys.path.insert(0, BASE)

from external_validation import (
    ELEMENTS, RADII, SHEAR_MOD, EN, TM, VEC_VALS, BULK_MOD, A_FCC, MASS, HMIX,
    R_GAS, compute_oliynyk_features, compute_hea_descriptors,
    compute_sisso_inputs, load_all_external_data,
)

t0 = time.time()

print("=" * 70)
print("ROBUST SISSO MODEL SEARCH")
print("=" * 70)


# ============================================================
# SECTION 0: LOAD TRAINING DATA + COMPUTE FEATURES
# ============================================================

DATA_CSV = f'{DATA_DIR}/data_with_vlc.csv'
df = pd.read_csv(DATA_CSV)
if 'eps_Labusch.1' in df.columns:
    df = df.drop(columns=['eps_Labusch.1'])
df['Omega'] = df['Omega'].clip(upper=100)
df = df.replace([np.inf, -np.inf], np.nan)

df_ys = df.dropna(subset=['YS']).copy()
y = df_ys['YS'].values
n = len(y)
d = df_ys['GrainSize'].values
groups = df_ys['Iteration'].values

print(f"Training data: {n} alloys, YS range {y.min():.0f}-{y.max():.0f} MPa")

# Compute Oliynyk features
def compute_oliynyk_row(row):
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

oliynyk_df = df_ys.apply(compute_oliynyk_row, axis=1)
for col in oliynyk_df.columns:
    df_ys[col] = oliynyk_df[col].values

# HEA descriptors already in data_with_vlc.csv
# Add Toda-Caraballo SSS if not present
if 'sigma_TC' not in df_ys.columns:
    df_ys['sigma_TC'] = 0.0

# Grain size feature
df_ys['d_inv_sqrt'] = d ** (-0.5)

print(f"  Computed {len(oliynyk_df.columns)} Oliynyk features")


# ============================================================
# SECTION 1: FEATURE SUBSET DEFINITIONS
# ============================================================

# Base feature set (matches sisso_analysis.py:176-193)
FEAT_BASE = [
    'r_mean', 'r_var', 'r_delta', 'r_range',
    'mu_mean', 'mu_var', 'mu_delta', 'mu_range',
    'EN_mean', 'EN_var', 'EN_delta', 'EN_range',
    'Tm_mean', 'Tm_var', 'Tm_delta', 'Tm_range',
    'VEC_mean',
    'K_mean', 'K_var',
    'delta', 'dS_mix', 'dH_mix', 'Omega',
    'Phi_VLC', 'eps_Labusch',
    'sigma_TC',
    'd_inv_sqrt',
]

# Check delta_mu vs mu_delta in data
if 'delta_mu' in df_ys.columns:
    FEAT_BASE.insert(FEAT_BASE.index('delta'), 'delta_mu')
    print(f"  delta_mu found in data (from eda descriptors)")

VARIANTS = {
    'v1_baseline': {
        'features': list(FEAT_BASE),
        'description': 'Original 30 features (baseline)',
    },
    'no_delta_mu': {
        'features': [f for f in FEAT_BASE if f not in ('delta_mu', 'mu_delta')],
        'description': 'Remove delta_mu and mu_delta',
    },
    'no_deltas': {
        'features': [f for f in FEAT_BASE if f not in (
            'delta_mu', 'mu_delta', 'delta', 'r_delta', 'EN_delta', 'Tm_delta')],
        'description': 'Remove all delta-type mismatch features',
    },
    'swap_mu_var': {
        'features': [('mu_var' if f == 'delta_mu' else f) for f in FEAT_BASE
                      if f != 'mu_delta'],
        'description': 'Replace delta_mu with mu_var (bounded)',
    },
}

# Deduplicate features
for k, v in VARIANTS.items():
    v['features'] = list(dict.fromkeys(v['features']))

for name, v in VARIANTS.items():
    print(f"  {name}: {len(v['features'])} features — {v['description']}")


# ============================================================
# SECTION 2: HELPER FUNCTIONS (from sisso_analysis.py)
# ============================================================

def compute_ic(y_true, y_pred, k, n):
    """AIC, AICc, BIC for a model with k parameters."""
    rss = np.sum((y_true - y_pred) ** 2)
    ll = n * np.log(rss / n)
    aic = ll + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    bic = ll + k * np.log(n)
    return {'AIC': aic, 'AICc': aicc, 'BIC': bic}


def run_sisso(df_input, operators, n_operators, dimension, sis_features, label=""):
    """Run SISSO feature expansion + regression."""
    print(f"\n  Running SISSO ({label})...")
    t_start = time.time()

    fsc = feature_space_construction(
        operators=operators, df=df_input.copy(),
        no_of_operators=n_operators, device='cpu'
    )
    x_expanded, y_tensor, names = fsc.feature_space()
    print(f"    Expanded: {x_expanded.shape[1]} features")

    reg = Regressor(x_expanded, y_tensor, names,
                    dimension=dimension, sis_features=sis_features)
    rmse, equation, r2, equations_per_dim = reg.regressor_fit()

    elapsed = time.time() - t_start
    print(f"    Equation: {equation}")
    print(f"    Train R²: {r2:.4f}, Time: {elapsed:.1f}s")

    return {
        'equation': equation, 'r2': r2, 'rmse': rmse,
        'x_expanded': x_expanded, 'y_tensor': y_tensor,
        'names': names, 'time': elapsed,
    }


def eval_sisso_equation(equation_str, x_data, names):
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
        matched = False
        if feat_name in name_to_idx:
            prediction += coef * x_np[:, name_to_idx[feat_name]]
            matched = True
        else:
            for name, idx in name_to_idx.items():
                if name.strip() == feat_name:
                    prediction += coef * x_np[:, idx]
                    matched = True
                    break
        if not matched:
            print(f"    WARNING: feature '{feat_name}' not found")

    remaining = re.sub(term_pattern, '', eq).strip()
    if remaining:
        try:
            prediction += float(remaining.replace(' ', ''))
        except ValueError:
            pass

    return prediction[0] if prediction.shape[0] == 1 else prediction


def sisso_loo(x_expanded, y_tensor, names, dimension, sis_features):
    """LOO-CV with pre-expanded feature space."""
    n_samples = len(y_tensor)
    preds = np.zeros(n_samples)
    x_np = x_expanded.numpy() if isinstance(x_expanded, torch.Tensor) else x_expanded
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        x_train = torch.tensor(x_np[mask], dtype=torch.float32)
        y_train = torch.tensor(y_np[mask], dtype=torch.float32)

        reg = Regressor(x_train, y_train, names,
                        dimension=dimension, sis_features=sis_features)
        rmse, equation, r2, _ = reg.regressor_fit()
        preds[i] = eval_sisso_equation(equation, x_np[i:i+1], names)

    r2_loo = r2_score(y_np, preds)
    rmse_loo = np.sqrt(mean_squared_error(y_np, preds))
    mae_loo = mean_absolute_error(y_np, preds)
    return r2_loo, rmse_loo, mae_loo, preds


def sisso_lobo(x_expanded, y_tensor, names, groups_arr, dimension, sis_features):
    """Leave-one-batch/cluster-out CV with pre-expanded feature space.

    Mirrors sisso_loo but holds out an entire Iteration batch per fold, so the
    robust variant reports the same aggregate LOBO metric that CLAUDE.md s11
    treats as mandatory for any cross-validated row.
    """
    x_np = x_expanded.numpy() if isinstance(x_expanded, torch.Tensor) else x_expanded
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor
    preds = np.zeros(len(y_np))
    for tr, te in LeaveOneGroupOut().split(x_np, y_np, groups_arr):
        reg = Regressor(torch.tensor(x_np[tr], dtype=torch.float32),
                        torch.tensor(y_np[tr], dtype=torch.float32),
                        names, dimension=dimension, sis_features=sis_features)
        _, equation, _, _ = reg.regressor_fit()
        preds[te] = eval_sisso_equation(equation, x_np[te], names)
    return r2_score(y_np, preds)


# ============================================================
# SECTION 2: RUN SISSO VARIANTS
# ============================================================
print("\n" + "=" * 70)
print("RUNNING SISSO VARIANTS")
print("=" * 70)

OPERATORS = ['+', '-', '*', '/']
N_OPERATORS = 2
DIMENSION = 3
SIS_FEATURES = 20

results = {}

for variant_name, variant_info in VARIANTS.items():
    print(f"\n{'='*50}")
    print(f"VARIANT: {variant_name}")
    print(f"{'='*50}")

    feat_list = variant_info['features']

    # Build DataFrame
    df_sisso = pd.DataFrame()
    df_sisso['YS'] = y
    for feat in feat_list:
        if feat in df_ys.columns:
            vals = df_ys[feat].fillna(0).values
            if np.std(vals) > 1e-12:
                df_sisso[feat] = vals
            else:
                print(f"    Skipping zero-variance: {feat}")
        else:
            print(f"    WARNING: feature '{feat}' not in data")

    n_feats = len(df_sisso.columns) - 1
    print(f"  Using {n_feats} features (after filtering)")

    # Run SISSO
    result = run_sisso(df_sisso, OPERATORS, N_OPERATORS, DIMENSION, SIS_FEATURES,
                       label=variant_name)

    # Training predictions
    y_pred_train = eval_sisso_equation(
        result['equation'], result['x_expanded'], result['names'])
    k_eff = DIMENSION + 1
    ic = compute_ic(y, y_pred_train, k_eff, n)

    # LOO-CV
    print(f"  Computing LOO-CV ({n} folds)...")
    t_loo = time.time()
    r2_loo, rmse_loo, mae_loo, loo_preds = sisso_loo(
        result['x_expanded'], result['y_tensor'],
        result['names'], DIMENSION, SIS_FEATURES)
    loo_time = time.time() - t_loo
    print(f"    LOO R² = {r2_loo:.4f}, RMSE = {rmse_loo:.1f}, MAE = {mae_loo:.1f} ({loo_time:.0f}s)")

    # LOBO-CV (leave-one-batch/cluster-out) — mandatory per CLAUDE.md s11
    print(f"  Computing LOBO-CV ({len(set(groups))} folds)...")
    t_lobo = time.time()
    r2_lobo = sisso_lobo(result['x_expanded'], result['y_tensor'],
                         result['names'], groups, DIMENSION, SIS_FEATURES)
    lobo_time = time.time() - t_lobo
    print(f"    LOBO R² = {r2_lobo:.4f} ({lobo_time:.0f}s)")

    results[variant_name] = {
        'equation': result['equation'],
        'train_r2': result['r2'],
        'loo_r2': r2_loo,
        'loo_rmse': rmse_loo,
        'loo_mae': mae_loo,
        'lobo_r2': r2_lobo,
        'bic': ic['BIC'],
        'k_eff': k_eff,
        'n_features': n_feats,
        'x_expanded': result['x_expanded'],
        'y_tensor': result['y_tensor'],
        'names': result['names'],
        'time': result['time'] + loo_time,
    }

# Add v2_dim3 (pre-computed, no re-run)
print(f"\n{'='*50}")
print("VARIANT: v2_dim3 (pre-computed)")
print(f"{'='*50}")

v2_bic = pd.read_csv(f'{RESULTS_DIR}/sisso_v2_bic_landscape.csv')
v2_eq = v2_bic.loc[v2_bic['dim'] == 3, 'equation'].values[0]
v2_train_r2 = v2_bic.loc[v2_bic['dim'] == 3, 'Train_R2'].values[0]
v2_bic_val = v2_bic.loc[v2_bic['dim'] == 3, 'BIC'].values[0]

# Read pre-computed LOO from sisso_v2_results.csv
v2_results = pd.read_csv(f'{RESULTS_DIR}/sisso_v2_results.csv')
v2_loo_r2 = v2_results['LOO_R2'].values[0]
v2_loo_rmse = v2_results['LOO_RMSE'].values[0]
v2_loo_mae = v2_results['LOO_MAE'].values[0]
v2_lobo_r2 = v2_results['LOBO_R2'].values[0] if 'LOBO_R2' in v2_results.columns else np.nan

print(f"  Equation: {v2_eq}")
print(f"  Train R² = {v2_train_r2:.4f}, LOO R² = {v2_loo_r2:.4f}, LOBO R² = {v2_lobo_r2:.4f}")

results['v2_dim3'] = {
    'equation': v2_eq,
    'train_r2': v2_train_r2,
    'loo_r2': v2_loo_r2,
    'loo_rmse': v2_loo_rmse,
    'loo_mae': v2_loo_mae,
    'lobo_r2': v2_lobo_r2,
    'bic': v2_bic_val,
    'k_eff': 4,
    'n_features': 'N/A (v2)',
    'time': 0,
}


# ============================================================
# SECTION 3: EXTERNAL VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("EXTERNAL VALIDATION")
print("=" * 70)

# Load external data
ext_models = {}
ext_models_temp = {}

# Fit models for external prediction
# For each variant, we need to refit the equation on ALL training data
# using the SISSO-discovered features, then predict on external data

for variant_name, res in results.items():
    if variant_name == 'v2_dim3':
        continue  # Handle separately

    # Extract the SISSO equation features and refit OLS
    equation = res['equation']
    x_exp = res['x_expanded']
    names = res['names']

    # Get training predictions for the full-data fit
    y_pred_full = eval_sisso_equation(equation, x_exp, names)

    # For external prediction, we need to:
    # 1. Parse which composite features the equation uses
    # 2. Compute those features for external data
    # Since this requires reproducing the feature expansion pipeline,
    # we'll use a simpler approach: refit OLS on the 3 SISSO terms

    # Extract the 3 term features from the equation
    term_pattern = r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*(.+?)(?=\s*[+-]\s*[\d.]|\s*$)'
    terms = re.findall(term_pattern, equation)

    name_to_idx = {name: i for i, name in enumerate(names)}
    x_np = x_exp.numpy() if isinstance(x_exp, torch.Tensor) else x_exp

    term_indices = []
    term_names = []
    for coef_str, feat_name in terms:
        feat_name = feat_name.strip()
        if feat_name in name_to_idx:
            term_indices.append(name_to_idx[feat_name])
            term_names.append(feat_name)
        else:
            for name, idx in name_to_idx.items():
                if name.strip() == feat_name:
                    term_indices.append(idx)
                    term_names.append(feat_name)
                    break

    if len(term_indices) == DIMENSION:
        X_terms = x_np[:, term_indices]
        reg_ols = LinearRegression().fit(X_terms, y)
        ext_models[variant_name] = {
            'ols': reg_ols,
            'term_names': term_names,
            'coefs': reg_ols.coef_,
            'intercept': reg_ols.intercept_,
        }
        print(f"\n  {variant_name}: extracted {len(term_names)} terms")
        for i, tn in enumerate(term_names):
            print(f"    Term {i+1}: {reg_ols.coef_[i]:.4f} * {tn}")
        print(f"    Intercept: {reg_ols.intercept_:.4f}")
    else:
        print(f"  WARNING: {variant_name} — could not extract {DIMENSION} terms")


# ============================================================
# SECTION 4: ROBUSTNESS STRESS TEST (all 56 ternaries)
# ============================================================

def _evaluate_composite_feature(term_name, feat_dict):
    """Evaluate a composite SISSO feature like 'EN_var/delta_mu' from a dict."""
    tn = term_name.strip()

    # Try direct lookup
    if tn in feat_dict:
        return feat_dict[tn]

    # Try binary operators: a/b, a*b, a+b, a-b
    for op in ['/', '*', '+', '-']:
        if op in tn:
            parts = tn.split(op, 1)
            if len(parts) == 2:
                left = parts[0].strip().strip('(')
                right = parts[1].strip().strip(')')

                left_val = feat_dict.get(left)
                right_val = feat_dict.get(right)

                if left_val is not None and right_val is not None:
                    if op == '/':
                        return left_val / right_val if abs(right_val) > 1e-15 else 0
                    elif op == '*':
                        return left_val * right_val
                    elif op == '+':
                        return left_val + right_val
                    elif op == '-':
                        return left_val - right_val

    return None


print("\n" + "=" * 70)
print("ROBUSTNESS STRESS TEST (56 equiatomic ternaries at d=50 μm)")
print("=" * 70)

ternary_combos = list(combinations(ELEMENTS, 3))
gs_test = 50.0

for variant_name, res in results.items():
    equation = res['equation']

    if variant_name == 'v2_dim3':
        # Need to evaluate v2 features — uses ln_d, Omega, etc.
        # Parse the equation manually
        n_unphysical = 0
        max_pred = -np.inf
        min_pred = np.inf
        worst_ternary = ""

        for trio in ternary_combos:
            fracs = {el: 1/3 if el in trio else 0 for el in ELEMENTS}
            oliynyk = compute_oliynyk_features(fracs)
            desc = compute_hea_descriptors(fracs)

            # v2 dim=3: 15071.43*(EN_var/Omega) + 53.66*(VEC_mean-ln_d)
            #          + 38250.04*(r_delta/r_range) - 142.20
            omega = desc['Omega']
            if omega > 100:
                omega = 100
            en_var = oliynyk['EN_var']
            vec_mean = oliynyk['VEC_mean']
            r_delta = oliynyk['r_delta']
            r_range = oliynyk['r_range']
            ln_d = np.log(gs_test)

            f1 = en_var / omega if omega > 0.01 else 0
            f2 = vec_mean - ln_d
            f3 = r_delta / r_range if r_range > 0 else 0

            pred = 15071.43 * f1 + 53.66 * f2 + 38250.04 * f3 - 142.20

            if pred > max_pred:
                max_pred = pred
                if pred > 1500:
                    worst_ternary = "-".join(trio)
            if pred < min_pred:
                min_pred = pred

            if pred > 1500 or pred < -500:
                n_unphysical += 1

        results[variant_name]['n_unphysical'] = n_unphysical
        results[variant_name]['pred_range'] = (min_pred, max_pred)
        print(f"  {variant_name}: range [{min_pred:.0f}, {max_pred:.0f}], unphysical: {n_unphysical}")
        continue

    # For SISSO variants with expanded features, compute features for ternaries
    # by reconstructing the feature expansion
    x_exp = res['x_expanded']
    names_list = res['names']

    # We can't easily evaluate expanded features for new compositions
    # without re-running feature_space_construction.
    # Instead, use the extracted OLS models from Section 3 if available.

    if variant_name not in ext_models:
        print(f"  {variant_name}: skipping (no OLS model extracted)")
        results[variant_name]['n_unphysical'] = -1
        results[variant_name]['pred_range'] = (0, 0)
        continue

    model_info = ext_models[variant_name]
    term_names = model_info['term_names']
    ols = model_info['ols']

    n_unphysical = 0
    max_pred = -np.inf
    min_pred = np.inf
    worst_ternary = ""

    for trio in ternary_combos:
        fracs = {el: 1/3 if el in trio else 0 for el in ELEMENTS}
        oliynyk = compute_oliynyk_features(fracs)
        desc = compute_hea_descriptors(fracs)

        # Build feature dict with all possible features
        feat_dict = {}
        feat_dict.update(oliynyk)
        feat_dict.update(desc)
        feat_dict['d_inv_sqrt'] = gs_test ** (-0.5)

        # Evaluate each term
        term_vals = []
        can_evaluate = True
        for tn in term_names:
            val = _evaluate_composite_feature(tn, feat_dict)
            if val is None:
                can_evaluate = False
                break
            term_vals.append(val)

        if not can_evaluate:
            continue

        X = np.array([term_vals])
        pred = ols.predict(X)[0]

        if pred > max_pred:
            max_pred = pred
            if pred > 1500:
                worst_ternary = "-".join(trio)
        if pred < min_pred:
            min_pred = pred

        if pred > 1500 or pred < -500:
            n_unphysical += 1

    results[variant_name]['n_unphysical'] = n_unphysical
    results[variant_name]['pred_range'] = (min_pred, max_pred)
    flag = " *** SINGULARITY" if n_unphysical > 0 else " OK"
    print(f"  {variant_name}: range [{min_pred:.0f}, {max_pred:.0f}], unphysical: {n_unphysical}{flag}")


# ============================================================
# SECTION 5: COMPARISON TABLE + OUTPUT
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

print(f"\n  {'Variant':<16s} {'LOO_R²':>8s} {'LOO_RMSE':>9s} {'BIC':>8s} {'#Unphys':>8s} {'Time':>7s}")
print("  " + "-" * 60)

for name, res in results.items():
    n_unphys = res.get('n_unphysical', -1)
    unphys_str = f"{n_unphys}" if n_unphys >= 0 else "N/A"
    t = res.get('time', 0)
    print(f"  {name:<16s} {res['loo_r2']:8.4f} {res['loo_rmse']:9.1f} {res['bic']:8.1f} {unphys_str:>8s} {t:7.0f}s")

print(f"\n  Equations:")
for name, res in results.items():
    eq = res['equation']
    if len(eq) > 100:
        eq = eq[:97] + "..."
    print(f"    {name}: {eq}")

# Recommend best
robust_variants = {k: v for k, v in results.items()
                   if v.get('n_unphysical', -1) == 0}

if robust_variants:
    best_name = max(robust_variants, key=lambda k: robust_variants[k]['loo_r2'])
    best = robust_variants[best_name]
    print(f"\n  RECOMMENDED: {best_name}")
    print(f"    LOO R² = {best['loo_r2']:.4f}, BIC = {best['bic']:.1f}")
    print(f"    Equation: {best['equation']}")
else:
    print("\n  WARNING: No variant achieved 0 unphysical predictions")
    best_name = min(results, key=lambda k: results[k].get('n_unphysical', 999))
    print(f"  Best compromise: {best_name} ({results[best_name].get('n_unphysical', -1)} unphysical)")

# Save comparison CSV
comp_rows = []
for name, res in results.items():
    comp_rows.append({
        'variant': name,
        'equation': res['equation'],
        'train_r2': res['train_r2'],
        'loo_r2': res['loo_r2'],
        'loo_rmse': res['loo_rmse'],
        'lobo_r2': res.get('lobo_r2', np.nan),
        'bic': res['bic'],
        'n_unphysical': res.get('n_unphysical', -1),
    })

pd.DataFrame(comp_rows).to_csv(f'{RESULTS_DIR}/sisso_robust_comparison.csv', index=False)
print(f"\n  Saved: {RESULTS_DIR}/sisso_robust_comparison.csv")

elapsed = time.time() - t0
print(f"\n  Total time: {elapsed:.1f}s")
print("=" * 70)
print("DONE")
print("=" * 70)
