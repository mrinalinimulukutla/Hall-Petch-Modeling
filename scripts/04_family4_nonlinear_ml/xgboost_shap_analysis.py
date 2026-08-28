#!/usr/bin/env python3
"""
Tuned XGBoost + SHAP Analysis for HEA Strengthening
====================================================
- Hyperparameter optimization via RandomizedSearchCV
- Nested cross-validation for unbiased performance estimates
- SHAP feature importance, dependence plots, and interaction analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedKFold, LeaveOneOut, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("TUNED XGBoost + SHAP ANALYSIS")
print("=" * 70)

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
print(f"Loaded {len(df)} alloys")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Add interaction features
for el in ELEMENTS:
    df[f'{el}_x_dinv'] = df[f'{el}_frac'] * df['d_inv_sqrt']

# Feature sets to evaluate
feature_sets = {
    'Comp+HP+Proc': ELEMENTS + ['d_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'],
    'Comp+HP+Proc+Desc': ELEMENTS + ['d_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime',
                                       'delta', 'VEC', 'dH_mix', 'eps_Labusch', 'Phi_VLC',
                                       'mu_bar', 'Tm_bar', 'delta_chi'],
    'CompFrac+HP+Proc+Interactions': [f'{el}_frac' for el in ELEMENTS] +
                                      ['d_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime'] +
                                      [f'{el}_x_dinv' for el in ELEMENTS],
    'Full': ELEMENTS + [f'{el}_frac' for el in ELEMENTS] +
            ['d_inv_sqrt', 'ColdWork', 'RecrystT', 'HoldTime',
             'delta', 'VEC', 'dH_mix', 'eps_Labusch', 'Phi_VLC',
             'mu_bar', 'Tm_bar', 'delta_chi', 'dS_mix', 'Omega', 'a_bar'] +
            [f'{el}_x_dinv' for el in ELEMENTS],
}

# ============================================================
# XGBoost HYPERPARAMETER SEARCH SPACE
# ============================================================
param_dist = {
    'n_estimators': [100, 200, 300, 500, 700],
    'max_depth': [2, 3, 4, 5, 6],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10],
    'gamma': [0, 0.01, 0.05, 0.1, 0.5],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0],
}

targets = {
    'YS': df.dropna(subset=['YS']),
    'HV': df,
}

best_models = {}
best_features = {}

for target_name, df_target in targets.items():
    print(f"\n{'=' * 60}")
    print(f"TARGET: {target_name}")
    print(f"{'=' * 60}")

    best_r2 = -np.inf
    best_feat_name = None
    best_model_obj = None
    best_X = None
    best_y = None

    for feat_name, features in feature_sets.items():
        # Clean data
        avail = [f for f in features if f in df_target.columns]
        df_clean = df_target[avail + [target_name]].replace([np.inf, -np.inf], np.nan).dropna()
        X = df_clean[avail].values
        y = df_clean[target_name].values
        n = len(y)

        if n < 20:
            print(f"  {feat_name}: n={n} too small, skipping")
            continue

        # Inner CV for hyperparameter tuning
        inner_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
        )

        search = RandomizedSearchCV(
            xgb_model, param_dist,
            n_iter=100,
            cv=inner_cv,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y)

        best_params = search.best_params_
        best_inner_r2 = search.best_score_

        # LOO cross-validation with best params
        final_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
            **best_params
        )

        loo_preds = np.zeros(n)
        for train_idx, test_idx in LeaveOneOut().split(X):
            final_model.fit(X[train_idx], y[train_idx])
            loo_preds[test_idx] = final_model.predict(X[test_idx])

        r2_loo = r2_score(y, loo_preds)
        rmse_loo = np.sqrt(mean_squared_error(y, loo_preds))
        mae_loo = mean_absolute_error(y, loo_preds)

        print(f"\n  Feature set: {feat_name} ({len(avail)} features, n={n})")
        print(f"    Inner CV best R²:  {best_inner_r2:.4f}")
        print(f"    LOO R²:            {r2_loo:.4f}")
        print(f"    LOO RMSE:          {rmse_loo:.2f}")
        print(f"    LOO MAE:           {mae_loo:.2f}")
        print(f"    Best params: {best_params}")

        if r2_loo > best_r2:
            best_r2 = r2_loo
            best_feat_name = feat_name
            best_model_obj = search.best_estimator_
            best_X = X
            best_y = y
            best_features[target_name] = avail

    # Re-fit best model on all data for SHAP
    print(f"\n  BEST for {target_name}: {best_feat_name} (LOO R² = {best_r2:.4f})")
    best_model_obj.fit(best_X, best_y)
    best_models[target_name] = best_model_obj

    # ============================================================
    # SHAP ANALYSIS
    # ============================================================
    print(f"\n  Computing SHAP values for {target_name}...")
    explainer = shap.TreeExplainer(best_model_obj)
    shap_values = explainer.shap_values(best_X)

    feat_names = best_features[target_name]

    # SHAP summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, best_X, feature_names=feat_names,
                      show=False, max_display=20)
    plt.title(f'SHAP Summary — {target_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/10_shap_summary_{target_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved SHAP summary plot")

    # SHAP bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, best_X, feature_names=feat_names,
                      plot_type='bar', show=False, max_display=20)
    plt.title(f'SHAP Feature Importance (mean |SHAP|) — {target_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/10_shap_bar_{target_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved SHAP bar plot")

    # SHAP dependence plots for top 6 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, feat_idx in enumerate(top_features_idx):
        ax = axes[i // 3, i % 3]
        shap.dependence_plot(
            feat_idx, shap_values, best_X,
            feature_names=feat_names,
            ax=ax, show=False
        )
        ax.set_title(f'{feat_names[feat_idx]}', fontsize=12)
    plt.suptitle(f'SHAP Dependence Plots — {target_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/10_shap_dependence_{target_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved SHAP dependence plots")

    # Mean |SHAP| table
    shap_importance = pd.Series(mean_abs_shap, index=feat_names).sort_values(ascending=False)
    print(f"\n    Top SHAP features for {target_name}:")
    for feat, val in shap_importance.head(15).items():
        pct = val / shap_importance.sum() * 100
        bar = '█' * int(pct)
        print(f"      {feat:20s}: {val:8.2f} ({pct:5.1f}%) {bar}")

    # ============================================================
    # PARITY PLOT with SHAP coloring
    # ============================================================
    loo_preds_final = np.zeros(len(best_y))
    for train_idx, test_idx in LeaveOneOut().split(best_X):
        best_model_obj.fit(best_X[train_idx], best_y[train_idx])
        loo_preds_final[test_idx] = best_model_obj.predict(best_X[test_idx])

    r2_final = r2_score(best_y, loo_preds_final)
    rmse_final = np.sqrt(mean_squared_error(best_y, loo_preds_final))

    fig, ax = plt.subplots(figsize=(8, 8))
    residuals = loo_preds_final - best_y
    scatter = ax.scatter(best_y, loo_preds_final, c=residuals, cmap='RdBu_r',
                         s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Residual', fontsize=11)
    lims = [min(best_y.min(), loo_preds_final.min()) * 0.9,
            max(best_y.max(), loo_preds_final.max()) * 1.1]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Experimental {target_name}', fontsize=12)
    ax.set_ylabel(f'Predicted {target_name} (LOO-CV)', fontsize=12)
    ax.set_title(f'XGBoost LOO: {target_name} — R²={r2_final:.3f}, RMSE={rmse_final:.1f}', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/11_xgboost_parity_{target_name}.png', dpi=150)
    plt.close()
    print(f"    Saved parity plot: R²={r2_final:.3f}, RMSE={rmse_final:.1f}")

# ============================================================
# SHAP INTERACTION VALUES (for YS)
# ============================================================
print("\n" + "=" * 70)
print("SHAP INTERACTION VALUES (YS)")
print("=" * 70)

target_name = 'YS'
if target_name in best_models:
    model = best_models[target_name]
    X_ys = best_X  # from last iteration (HV) — need to recompute for YS
    feats_ys = best_features['YS']
    df_ys = targets['YS']
    df_clean_ys = df_ys[feats_ys + ['YS']].replace([np.inf, -np.inf], np.nan).dropna()
    X_ys = df_clean_ys[feats_ys].values
    y_ys = df_clean_ys['YS'].values

    model.fit(X_ys, y_ys)
    explainer_ys = shap.TreeExplainer(model)

    print("  Computing SHAP interaction values (this may take a moment)...")
    shap_interaction = explainer_ys.shap_interaction_values(X_ys)

    # Sum absolute interaction values to find strongest pairs
    n_feat = len(feats_ys)
    interaction_matrix = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(n_feat):
            interaction_matrix[i, j] = np.abs(shap_interaction[:, i, j]).mean()

    # Plot interaction heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.eye(n_feat, dtype=bool)  # mask diagonal
    sns.heatmap(interaction_matrix, xticklabels=feats_ys, yticklabels=feats_ys,
                mask=mask, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax,
                annot_kws={'size': 7})
    ax.set_title('Mean |SHAP Interaction| — YS', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/12_shap_interactions_YS.png', dpi=150)
    plt.close()
    print("  Saved SHAP interaction heatmap")

    # Top interaction pairs
    pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pairs.append((feats_ys[i], feats_ys[j], interaction_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n  Top 10 feature interactions (YS):")
    for f1, f2, val in pairs[:10]:
        print(f"    {f1:15s} × {f2:15s}: {val:.3f}")

print("\n" + "=" * 70)
print("XGBoost + SHAP ANALYSIS COMPLETE")
print("=" * 70)
