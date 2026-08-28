#!/usr/bin/env python3
"""
EML Symbolic Regression for Hall-Petch HEA Yield Strength
=========================================================
Implements the EML operator eml(x,y) = exp(x) - ln(y) from
Odrzywołek (arXiv:2603.21852) as gradient-based trainable circuits
for yield strength prediction in FCC HEAs.

Binary EML trees with linear-combination leaves are optimized via
PyTorch + Adam. Complex128 arithmetic handles intermediate negative
values under log (as recommended in the paper).
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings('ignore')

# ============================================================
# PATHS AND CONSTANTS
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'
DATA_FILE = f'{DATA_DIR}/data_with_vlc.csv'
ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
R_GAS = 8.314  # J/(mol·K)

BATCH_COLORS = {'BBA': '#D55E00', 'BBB': '#0072B2', 'BBC': '#009E73',
                'CBA': '#CC79A7', 'CBB': '#E69F00', 'CBC': '#56B4E9'}

# Elemental properties (verified metallurgical constants)
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

# Selected physics features (covers SISSO equation terms + extra)
EML_FEATURES = ['r_var', 'r_range', 'EN_var', 'mu_delta',
                'dS_mix', 'd_inv_sqrt', 'delta', 'VEC_mean']


# ============================================================
# DATA LOADING AND FEATURE COMPUTATION
# ============================================================
def load_data():
    """Load dataset and compute Oliynyk-style physics features."""
    df = pd.read_csv(DATA_FILE)
    if 'eps_Labusch.1' in df.columns:
        df = df.drop(columns=['eps_Labusch.1'])
    df['Omega'] = df['Omega'].clip(upper=100)
    df = df.replace([np.inf, -np.inf], np.nan)
    df_ys = df.dropna(subset=['YS']).copy()

    fracs = {el: df_ys[f'{el}_frac'].values for el in ELEMENTS}

    def comp_mean(prop):
        return sum(fracs[el] * prop[el] for el in ELEMENTS)

    def comp_var(prop):
        mu = comp_mean(prop)
        return sum(fracs[el] * (prop[el] - mu)**2 for el in ELEMENTS)

    def comp_delta(prop):
        mu = comp_mean(prop)
        v = comp_var(prop)
        return np.sqrt(v) / np.where(np.abs(mu) > 1e-10, np.abs(mu), 1e-10)

    def comp_range(prop):
        n = len(df_ys)
        out = np.zeros(n)
        for i in range(n):
            vals = [prop[el] for el in ELEMENTS if fracs[el][i] > 0.001]
            if len(vals) >= 2:
                out[i] = max(vals) - min(vals)
        return out

    # Compute features
    df_ys['r_var'] = comp_var(RADII)
    df_ys['r_range'] = comp_range(RADII)
    df_ys['EN_var'] = comp_var(EN)
    df_ys['mu_delta'] = comp_delta(SHEAR_MOD)

    eps = 1e-12
    df_ys['dS_mix'] = -R_GAS * sum(
        np.where(fracs[el] > eps, fracs[el] * np.log(fracs[el] + eps), 0)
        for el in ELEMENTS)

    df_ys['d_inv_sqrt'] = df_ys['GrainSize'].values ** (-0.5)

    r_mean = comp_mean(RADII)
    df_ys['delta'] = np.sqrt(sum(
        fracs[el] * (1 - RADII[el] / r_mean)**2
        for el in ELEMENTS)) * 100

    df_ys['VEC_mean'] = comp_mean(VEC_VALS)

    X = df_ys[EML_FEATURES].values.astype(np.float64)
    y = df_ys['YS'].values.astype(np.float64)
    groups = df_ys['Iteration'].values

    print(f"Loaded {len(y)} alloys, {len(EML_FEATURES)} features")
    print(f"YS range: {y.min():.0f}–{y.max():.0f} MPa (mean={y.mean():.0f})")
    return X, y, groups


# ============================================================
# EML TREE MODEL
# ============================================================
class EMLTree(nn.Module):
    """Complete binary tree of EML operators with linear leaves.

    eml(x, y) = exp(x) - ln(y)

    Uses complex128 arithmetic internally to handle ln(negative).
    Final output is real-valued via affine transformation.
    """

    def __init__(self, n_features, depth=2, y_mean=0.0, y_std=1.0):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.n_leaves = 2 ** depth

        # Leaf parameters: leaf_i = bias_i + Σ weight_ij · x_j
        self.W = nn.Parameter(torch.randn(self.n_leaves, n_features,
                                          dtype=torch.float64) * 0.1)
        self.b = nn.Parameter(torch.randn(self.n_leaves,
                                          dtype=torch.float64) * 0.3)

        # Output affine: y_pred = scale * Re(tree) + offset
        self.scale = nn.Parameter(torch.tensor(y_std / 2, dtype=torch.float64))
        self.offset = nn.Parameter(torch.tensor(y_mean, dtype=torch.float64))

    def forward(self, X):
        """X: (batch, n_features) real → predictions (batch,) real."""
        # Compute leaf values (real) then cast to complex
        leaves_real = X @ self.W.T + self.b          # (batch, n_leaves)
        vals = leaves_real.to(torch.complex128)       # complex leaves

        # Bottom-up EML evaluation
        for _ in range(self.depth):
            left = vals[:, 0::2]
            right = vals[:, 1::2]
            # eml(left, right) = exp(left) - ln(right)
            left_clamped = torch.complex(
                torch.clamp(left.real, -15, 15), left.imag)
            exp_left = torch.exp(left_clamped)
            # Safe log: floor magnitude to avoid log(0)
            mag = torch.abs(right)
            safe_right = torch.where(mag < 1e-30,
                                     torch.full_like(right, 1e-30),
                                     right)
            ln_right = torch.log(safe_right)
            vals = exp_left - ln_right

        # Extract real part, apply affine
        tree_out = vals.squeeze(-1).real
        return self.scale * tree_out + self.offset

    def count_params(self, threshold=0.0):
        """Count non-zero parameters (after optional pruning)."""
        n = 2  # scale + offset
        W = self.W.detach().abs()
        b = self.b.detach().abs()
        n += (W > threshold).sum().item()
        n += (b > threshold).sum().item()
        return n

    def feature_importance(self, feat_names):
        """Total absolute weight per feature across all leaves."""
        W = self.W.detach().cpu().numpy()
        importance = np.abs(W).sum(axis=0)
        pairs = sorted(zip(feat_names, importance),
                       key=lambda x: -x[1])
        return pairs

    def get_expression(self, feat_names, threshold=0.05):
        """Extract human-readable EML expression."""
        W = self.W.detach().cpu().numpy()
        b = self.b.detach().cpu().numpy()
        scale = self.scale.item()
        offset = self.offset.item()

        def leaf_str(idx):
            terms = []
            for j, name in enumerate(feat_names):
                if abs(W[idx, j]) > threshold:
                    terms.append(f'{W[idx, j]:+.3f}·{name}')
            bias = b[idx]
            if abs(bias) > threshold or not terms:
                terms.append(f'{bias:+.3f}')
            return '(' + ' '.join(terms) + ')'

        def build(nodes):
            if len(nodes) == 1:
                return nodes[0]
            left = nodes[0::2]
            right = nodes[1::2]
            merged = [f'eml({l}, {r})' for l, r in zip(left, right)]
            return build(merged)

        leaves = [leaf_str(i) for i in range(self.n_leaves)]
        tree = build(leaves)
        return f'{scale:.3f} · Re({tree}) + {offset:.3f}'


# ============================================================
# TRAINING
# ============================================================
def train_eml(X_train, y_train, depth, n_restarts=50, n_epochs=5000,
              lr=1e-3, wd=1e-4, verbose=False):
    """Train EML tree with multiple random restarts.

    Returns best model and its training predictions.
    """
    n, p = X_train.shape
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    y_mean, y_std = y_train.mean(), max(y_train.std(), 1.0)

    best_model_state = None
    best_loss = float('inf')

    for restart in range(n_restarts):
        model = EMLTree(p, depth=depth, y_mean=y_mean, y_std=y_std)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = model(X_t)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                break  # abandon this restart

            loss = nn.MSELoss()(pred, y_t)
            if torch.isnan(loss):
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

        # Evaluate final loss (if training didn't break)
        with torch.no_grad():
            pred = model(X_t)
            if not (torch.isnan(pred).any() or torch.isinf(pred).any()):
                final_loss = nn.MSELoss()(pred, y_t).item()
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_model_state = {k: v.clone()
                                        for k, v in model.state_dict().items()}
                    if verbose and (restart + 1) % 20 == 0:
                        r2 = 1 - final_loss / np.var(y_train)
                        print(f'    restart {restart+1}/{n_restarts}: '
                              f'loss={final_loss:.1f}, R²={r2:.3f}')

    if best_model_state is None:
        print(f'  WARNING: all {n_restarts} restarts diverged at depth {depth}')
        return None, float('inf'), np.full(n, np.nan)

    # Rebuild best model
    model = EMLTree(p, depth=depth, y_mean=y_mean, y_std=y_std)
    model.load_state_dict(best_model_state)
    with torch.no_grad():
        preds = model(X_t).numpy()

    return model, best_loss, preds


# ============================================================
# IC COMPUTATION
# ============================================================
def compute_ic(y_true, y_pred, k, n):
    """Compute AIC, AICc, BIC."""
    rss = np.sum((y_true - y_pred) ** 2)
    if rss <= 0:
        rss = 1e-15
    log_term = n * np.log(rss / n)
    aic = log_term + 2 * k
    bic = log_term + k * np.log(n)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf
    return aic, aicc, bic


# ============================================================
# PARITY PLOT
# ============================================================
def parity_plot(ax, y_true, y_pred, groups, title):
    """Parity plot colored by batch."""
    for batch, color in BATCH_COLORS.items():
        mask = groups == batch
        if mask.any():
            ax.scatter(y_true[mask], y_pred[mask], c=color, s=50,
                       alpha=0.7, edgecolors='k', linewidth=0.5, label=batch)
    valid = ~np.isnan(y_pred)
    if valid.any():
        lo = min(y_true[valid].min(), y_pred[valid].min()) * 0.9
        hi = max(y_true[valid].max(), y_pred[valid].max()) * 1.1
    else:
        lo, hi = 100, 600
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Experimental YS (MPa)', fontsize=10)
    ax.set_ylabel('EML Predicted YS (MPa)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')


# ============================================================
# MAIN ANALYSIS
# ============================================================
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 70)
    print('EML SYMBOLIC REGRESSION — Hall-Petch HEA Yield Strength')
    print('Operator: eml(x,y) = exp(x) - ln(y)  [Odrzywołek 2025]')
    print('=' * 70)

    # --- Load data ---
    X_raw, y, groups = load_data()
    n = len(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # =========================================================
    # PHASE 1: Full-data training at depths 1, 2, 3
    # =========================================================
    print('\n--- Phase 1: Full-data training ---')
    results = {}
    best_depth = None
    best_train_r2 = -np.inf

    for depth in [1, 2, 3]:
        n_params = 2 ** depth * (len(EML_FEATURES) + 1) + 2
        print(f'\nDepth {depth}: {2**depth} leaves, {n_params} parameters')
        model, loss, preds = train_eml(X, y, depth=depth,
                                       n_restarts=100, n_epochs=5000,
                                       verbose=True)
        if model is None:
            results[depth] = {'model': None, 'train_R2': -999,
                              'preds': preds, 'k_eff': n_params}
            continue

        train_r2 = r2_score(y, preds)
        aic, aicc, bic = compute_ic(y, preds, n_params, n)
        k_pruned = model.count_params(threshold=0.05)

        results[depth] = {
            'model': model, 'train_R2': train_r2, 'preds': preds,
            'k_eff': n_params, 'k_pruned': k_pruned,
            'AIC': aic, 'AICc': aicc, 'BIC': bic,
        }

        print(f'  Train R² = {train_r2:.4f}  |  '
              f'AIC = {aic:.1f}  |  BIC = {bic:.1f}  |  '
              f'k_eff = {n_params} (pruned: {k_pruned})')

        # Feature importance
        importance = model.feature_importance(EML_FEATURES)
        top3 = ', '.join(f'{name}={imp:.2f}' for name, imp in importance[:3])
        print(f'  Top features: {top3}')

        # Expression
        expr = model.get_expression(EML_FEATURES)
        print(f'  Expression: {expr[:200]}')

        if train_r2 > best_train_r2:
            best_train_r2 = train_r2
            best_depth = depth

    print(f'\nBest depth by training R²: {best_depth} '
          f'(R²={best_train_r2:.4f})')

    # =========================================================
    # PHASE 2: LOO-CV for depths 1, 2, and optionally 3
    # =========================================================
    depths_for_loo = [1, 2]
    if best_depth == 3:
        depths_for_loo.append(3)

    loo_results = {}
    for depth in depths_for_loo:
        print(f'\n--- Phase 2: LOO-CV at depth {depth} ---')
        y_loo = np.zeros(n)
        n_failed = 0

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_tr, y_tr = X[mask], y[mask]

            model_i, _, _ = train_eml(X_tr, y_tr, depth=depth,
                                      n_restarts=20, n_epochs=3000)
            if model_i is None:
                y_loo[i] = y.mean()
                n_failed += 1
            else:
                with torch.no_grad():
                    X_te = torch.tensor(X[~mask], dtype=torch.float64)
                    y_loo[i] = model_i(X_te).item()

            if (i + 1) % 20 == 0:
                r2_so_far = r2_score(y[:i+1], y_loo[:i+1])
                print(f'  fold {i+1}/{n}  (running R²={r2_so_far:.3f})')

        r2_loo = r2_score(y, y_loo)
        rmse_loo = np.sqrt(mean_squared_error(y, y_loo))
        mae_loo = mean_absolute_error(y, y_loo)

        loo_results[depth] = {
            'R2': r2_loo, 'RMSE': rmse_loo, 'MAE': mae_loo,
            'preds': y_loo, 'failed': n_failed,
        }
        print(f'  Depth {depth} LOO: R²={r2_loo:.4f}, '
              f'RMSE={rmse_loo:.1f}, MAE={mae_loo:.1f}'
              f'{f" ({n_failed} failed)" if n_failed else ""}')

    # Select best LOO depth
    best_loo_depth = max(loo_results, key=lambda d: loo_results[d]['R2'])
    print(f'\nBest LOO depth: {best_loo_depth} '
          f'(R²={loo_results[best_loo_depth]["R2"]:.4f})')

    # =========================================================
    # PHASE 3: LOBO-CV for best LOO depth
    # =========================================================
    print(f'\n--- Phase 3: LOBO-CV at depth {best_loo_depth} ---')
    y_lobo = np.zeros(n)
    logo = LeaveOneGroupOut()
    for fold, (tr, te) in enumerate(logo.split(X, y, groups)):
        batch_name = groups[te[0]]
        model_b, _, _ = train_eml(X[tr], y[tr], depth=best_loo_depth,
                                  n_restarts=20, n_epochs=3000)
        if model_b is not None:
            with torch.no_grad():
                X_te = torch.tensor(X[te], dtype=torch.float64)
                y_lobo[te] = model_b(X_te).numpy()
        else:
            y_lobo[te] = y[tr].mean()
        print(f'  Batch {batch_name}: {len(te)} samples')

    r2_lobo = r2_score(y, y_lobo)
    print(f'  LOBO R² = {r2_lobo:.4f}')

    # =========================================================
    # PHASE 4: Summary and outputs
    # =========================================================
    print('\n' + '=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)

    summary_rows = []
    for depth in sorted(results.keys()):
        r = results[depth]
        if r['model'] is None:
            continue
        row = {
            'Depth': depth,
            'Leaves': 2 ** depth,
            'k_eff': r['k_eff'],
            'Train_R2': r['train_R2'],
            'AIC': r.get('AIC', np.nan),
            'BIC': r.get('BIC', np.nan),
        }
        if depth in loo_results:
            lr = loo_results[depth]
            row['LOO_R2'] = lr['R2']
            row['LOO_RMSE'] = lr['RMSE']
            row['LOO_MAE'] = lr['MAE']
        if depth == best_loo_depth:
            row['LOBO_R2'] = r2_lobo
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format='{:.4f}'.format))

    # --- Comparison with baselines ---
    print('\n--- Comparison ---')
    print(f'  SISSO Full:       LOO R²=0.672, BIC=714  (4 params)')
    print(f'  Stacking (Ridge): LOO R²=0.704, BIC=717  (6 params)')
    print(f'  XGBoost:          LOO R²=0.720, BIC=17156 (3806 params)')
    best_lr = loo_results[best_loo_depth]
    best_r = results[best_loo_depth]
    print(f'  EML depth-{best_loo_depth}:      '
          f'LOO R²={best_lr["R2"]:.3f}, '
          f'BIC={best_r.get("BIC", "N/A"):.0f}  '
          f'({best_r["k_eff"]} params)')

    # --- Best model expression ---
    best_model = results[best_loo_depth]['model']
    if best_model is not None:
        print(f'\nBest EML expression (depth {best_loo_depth}):')
        print(f'  {best_model.get_expression(EML_FEATURES)}')
        print(f'\nFeature importance:')
        for name, imp in best_model.feature_importance(EML_FEATURES):
            print(f'  {name:12s}: {imp:.4f}')

    # =========================================================
    # PHASE 5: Plots
    # =========================================================
    fig, axes = plt.subplots(1, len(loo_results), figsize=(6*len(loo_results), 6))
    if len(loo_results) == 1:
        axes = [axes]

    for ax, depth in zip(axes, sorted(loo_results.keys())):
        lr = loo_results[depth]
        r = results[depth]
        title = (f'EML depth-{depth} (LOO)\n'
                 f'R²={lr["R2"]:.3f}, RMSE={lr["RMSE"]:.1f} MPa\n'
                 f'k_eff={r["k_eff"]}')
        parity_plot(ax, y, lr['preds'], groups, title)

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/62_eml_parity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {PLOT_DIR}/62_eml_parity.png')

    # =========================================================
    # PHASE 6: Append to model_search_results_v2.csv
    # =========================================================
    csv_path = f'{RESULTS_DIR}/model_search_results_v2.csv'
    elapsed = time.time() - t0
    try:
        existing = pd.read_csv(csv_path)
        existing = existing[~existing['Model'].str.contains('EML', na=False)]

        eml_rows = []
        for depth in sorted(loo_results.keys()):
            lr = loo_results[depth]
            r = results[depth]
            eml_rows.append({
                'Model': f'EML (depth-{depth})',
                'Features': 'EML_physics',
                'n_feat': len(EML_FEATURES),
                'LOO_R2': lr['R2'],
                'LOO_RMSE': lr['RMSE'],
                'LOO_MAE': lr['MAE'],
                'LOBO_R2': r2_lobo if depth == best_loo_depth else np.nan,
                'Train_R2': r['train_R2'],
                'k_eff': r['k_eff'],
                'AIC': r.get('AIC', np.nan),
                'AICc': r.get('AICc', np.nan),
                'BIC': r.get('BIC', np.nan),
                'HPO': f'Adam-100restarts',
                'time': elapsed,
            })

        updated = pd.concat([existing, pd.DataFrame(eml_rows)],
                            ignore_index=True)
        updated = updated.sort_values('LOO_R2', ascending=False)
        updated.to_csv(csv_path, index=False)
        print(f'Updated: {csv_path}')
    except Exception as e:
        print(f'Warning: could not update CSV: {e}')

    # Save detailed results
    summary.to_csv(f'{RESULTS_DIR}/eml_results.csv', index=False)
    print(f'Saved: {RESULTS_DIR}/eml_results.csv')

    print(f'\nTotal time: {elapsed/60:.1f} min')
    print('Done.')
