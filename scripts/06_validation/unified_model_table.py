#!/usr/bin/env python3
"""
Unified Model-Selection Table
==============================
Builds the single master comparison table that the integrated paper / talk
will report. Every row is a candidate model. Every column is a metric.

Columns
-------
  Model                Display name
  Family               classical | linear | tree | symbolic
  Params               Effective parameter count
  LOO_R2               Leave-one-out cross-validated R²
  LOBO_R2              Leave-one-batch-out cross-validated R²
  BIC                  Bayesian Information Criterion
  Ext_RMSE_MPa         External-set RMSE on the 82-point literature set
  Singularity_safe     yes / partial / no — based on singularity_audit.csv
  Source               which results/*.csv produced the row

Inputs
------
  results/model_search_results.csv     (baseline: 17-model panel)
  results/sisso_results.csv            (baseline: SISSO Full + Robust + v2)
  results/external_tier_results.csv    (baseline: external aggregate RMSE)
  results/pca_ols_results.csv          (compact-equation stream)
  results/pysr_grid_summary_YS.csv     (compact-equation stream; pending PySR run)
  results/pysr_grid_summary_HV.csv     (compact-equation stream; pending PySR run)
  results/hardness_sr_results.csv      (compact-equation stream)
  results/singularity_audit.csv        (compact-equation stream)

Note: Bayesian PSIS-LOO output for the M0–M11 composition-HP hierarchy
lives in comp_hp_model_comparison.csv on a different scale (elpd_loo,
not LOO R²); it is reported separately in the paper §Model Comparison,
not merged into this unified-R² table.

Output
------
  results/unified_model_table.csv
  analysis_plots/83_unified_model_table.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import RESULTS_DIR, PLOTS_DIR

R = Path(RESULTS_DIR)

def maybe_read(path, **kw):
    p = R / path
    if not p.exists():
        print(f"[skip] missing: {p}")
        return None
    return pd.read_csv(p, **kw)

rows = []

# ---- Baseline stream: 17-model panel ------------------------------------------
# Schema: Model, Features, n_feat, LOO_R2, LOO_RMSE, LOO_MAE, LOBO_R2,
#         LOBO_RMSE, HPO, Time_s
ml_panel = maybe_read('model_search_results.csv')
if ml_panel is not None:
    for _, r in ml_panel.iterrows():
        name = str(r.get('Model', '?'))
        family = ('tree' if any(t in name for t in ('XGB', 'Forest', 'CatBoost', 'LightGBM', 'Tree'))
                  else 'linear')
        rows.append({
            'Model': name,
            'Family': family,
            'Params': r.get('n_feat', np.nan),
            'R2_5fold': r.get('R2_5fold', np.nan),
            'LOO_R2': r.get('LOO_R2', np.nan),
            'LOBO_R2': r.get('LOBO_R2', np.nan),
            'BIC': np.nan,
            'Ext_RMSE_MPa': np.nan,
            'Singularity_safe': 'n/a',
            'Source': 'model_search_results.csv',
        })

# ---- Baseline stream: SISSO Full + Robust (+ v2) ------------------------------
# Schema: Model, n_terms, LOO_R2, LOO_RMSE, LOO_MAE, LOBO_R2, Train_R2,
#         k_eff, AIC, AICc, BIC, Equation
sisso = maybe_read('sisso_results.csv')
if sisso is not None:
    for _, r in sisso.iterrows():
        name = str(r.get('Model', 'SISSO ?'))
        rows.append({
            'Model': name,
            'Family': 'symbolic',
            'Params': r.get('k_eff', 4),
            'R2_5fold': np.nan,
            'LOO_R2': r.get('LOO_R2', np.nan),
            'LOBO_R2': r.get('LOBO_R2', np.nan),
            'BIC': r.get('BIC', np.nan),
            'Ext_RMSE_MPa': np.nan,
            'Singularity_safe': 'no' if 'Full' in name else 'yes',
            'Source': 'sisso_results.csv',
        })

# ---- Baseline stream: external validation (Aggregate tier RMSE per model) -----
# Schema: label, R2, RMSE, MAE, bias, n. Label format: "Tier X / Model name"
ext = maybe_read('external_tier_results.csv')
if ext is not None:
    # Build explicit ext_model → ext_RMSE map (aggregate tier only)
    agg = ext[ext['label'].str.startswith('Aggregate')]
    ext_map = {}
    for _, r in agg.iterrows():
        if ' / ' in r['label']:
            ext_map[r['label'].split(' / ', 1)[1].strip()] = float(r['RMSE'])

    # Explicit row-name → ext_key map. Only models we *actually* ran on
    # the external set get a number; everything else stays NaN.
    EXT_ALIAS = {
        'SISSO Full': 'SISSO',
        'SISSO Robust': 'SISSO Robust',
        'M3: σ₀(all elem)': 'M3',
    }
    for row in rows:
        key = EXT_ALIAS.get(str(row['Model']))
        if key and key in ext_map:
            row['Ext_RMSE_MPa'] = ext_map[key]

# ---- Baseline stream: SISSO Robust (lives in its own CSV) ---------------------
# The canonical robust equation is the 'no_delta_mu' variant (paper Eq. 5);
# columns in this CSV are lowercase (loo_r2, lobo_r2, bic) — earlier code read
# uppercase keys off iloc[0] (= v1_baseline = SISSO Full), so LOO/BIC came back
# NaN/wrong. Select the correct variant and the correct column names.
sisso_rob = maybe_read('sisso_robust_comparison.csv')
if sisso_rob is not None and len(sisso_rob):
    rob = sisso_rob[sisso_rob['variant'] == 'no_delta_mu']
    r = rob.iloc[0] if len(rob) else sisso_rob.iloc[0]
    rows.append({
        'Model': 'SISSO Robust',
        'Family': 'symbolic',
        'Params': r.get('k_eff', 4),
        'R2_5fold': np.nan,
        'LOO_R2': r.get('loo_r2', np.nan),
        'LOBO_R2': r.get('lobo_r2', np.nan),
        'BIC': r.get('bic', np.nan),
        'Ext_RMSE_MPa': ext_map.get('SISSO Robust', np.nan) if ext is not None else np.nan,
        'Singularity_safe': 'yes',
        'Source': 'sisso_robust_comparison.csv',
    })

# ---- compact-equation stream: PCA-OLS ---------------------------------------------------
pca_ols = maybe_read('pca_ols_results.csv')
if pca_ols is not None:
    for _, r in pca_ols.iterrows():
        rows.append({
            'Model': r.get('model', 'PCA-OLS'),
            'Family': 'linear',
            'Params': int(r.get('n_pcs', 0)) + 1,
            'R2_5fold': r.get('R2_5fold', np.nan),
            'LOO_R2': r.get('R2_LOO', np.nan),
            'LOBO_R2': r.get('R2_LOBO', np.nan),
            'BIC': np.nan,
            'Ext_RMSE_MPa': np.nan,
            'Singularity_safe': 'yes',
            'Source': 'pca_ols_results.csv',
        })

# ---- compact-equation stream: PySR grid (YS and HV) — summarize the elbow rows ---------
# Schema (new, post-rewrite): target, feature_set, op_set, selection,
# complexity, n_constants, fit_R2,
# cv_refit_5fold_R2, cv_refit_LOO_R2, cv_refit_LOBO_R2,
# cv_frozen_5fold_R2, cv_frozen_LOO_R2, cv_frozen_LOBO_R2, equation
#
# BIC, external-set RMSE, and singularity_safe are filled from
# pysr_external_validation.csv (scripts/pysr_external_validation.py), which
# scores every PySR equation with the SAME BIC formula and the SAME 82-point
# external dataset that SISSO used. NOTE: PySR feature sets all use SD_GS
# (F2/F3 also use processing), none of which the external literature reports,
# so PySR external numbers are conditional on imputation (see imputed_features
# in that CSV). SISSO uses composition+grain only and needs no imputation.
pysr_ext = maybe_read('pysr_external_validation.csv')
ext_lookup = {}
if pysr_ext is not None:
    for _, e in pysr_ext.iterrows():
        ext_lookup[(e['Target'], e['feature_set'], e['op_set'], e['selection'])] = e
for tgt in ('YS', 'HV'):
    pysr_grid = maybe_read(f'pysr_grid_summary_{tgt}.csv')
    if pysr_grid is None:
        continue
    elbows = pysr_grid[pysr_grid['selection'] == 'elbow']
    for _, r in elbows.iterrows():
        cell_tag = f"{r['feature_set']}_{r['op_set']}"
        ev_row = ext_lookup.get((tgt, r['feature_set'], r['op_set'], 'elbow'))
        ext_rmse = np.nan
        if ev_row is not None and str(ev_row['Ext_RMSE_MPa']) not in ('', 'nan'):
            ext_rmse = ev_row['Ext_RMSE_MPa']
        rows.append({
            'Model': f"PySR-{tgt}-{cell_tag} (elbow)",
            'Family': 'symbolic',
            'Params': r['n_constants'] if pd.notna(r.get('n_constants')) else r['complexity'],
            'R2_5fold': r.get('cv_refit_5fold_R2', np.nan),
            'LOO_R2':   r.get('cv_refit_LOO_R2',   np.nan),
            'LOBO_R2':  r.get('cv_refit_LOBO_R2',  np.nan),
            'BIC': (ev_row['BIC'] if ev_row is not None else np.nan),
            'Ext_RMSE_MPa': ext_rmse,
            'Singularity_safe': (ev_row['Singularity_safe'] if ev_row is not None else 'TBD'),
            'Source': f'pysr_grid_summary_{tgt}.csv + pysr_external_validation.csv',
        })

# ---- SISSO + SD_grain (fair-comparison variant, YS + HV) ---------------
# scripts/sisso_with_sdgrain.py adds SD_GS to the SISSO SIS pool for both
# targets and external-validates on the SAME set/imputation as PySR. Kept
# separate from canonical SISSO (sisso_results.csv) so the paper's locked
# numbers are untouched.
sisso_sd = maybe_read('sisso_sdgrain_results.csv')
if sisso_sd is not None:
    for _, r in sisso_sd.iterrows():
        rows.append({
            'Model': r['Model'],
            'Family': 'symbolic',
            'Params': r['k_eff'],
            'R2_5fold': np.nan,
            'LOO_R2': r['LOO_R2'],
            'LOBO_R2': r['LOBO_R2'],
            'BIC': r['BIC'],
            'Ext_RMSE_MPa': (r['Ext_RMSE'] if str(r['Ext_RMSE']) not in ('', 'nan') else np.nan),
            'Singularity_safe': r['Singularity_safe'],
            'Source': 'sisso_sdgrain_results.csv',
        })

# ---- compact-equation stream: HV SR -----------------------------------------------------
hv_sr = maybe_read('hardness_sr_results.csv')
if hv_sr is not None:
    for _, r in hv_sr.iterrows():
        rows.append({
            'Model': f"HV-{r['equation']}",
            'Family': 'symbolic',
            'Params': 3,
            'R2_5fold': r.get('R2_5fold', np.nan),
            'LOO_R2': r.get('R2_LOO', np.nan),
            'LOBO_R2': r.get('R2_LOBO', np.nan),
            'BIC': np.nan,
            'Ext_RMSE_MPa': np.nan,
            'Singularity_safe': 'yes' if 'elbow' in str(r['equation']) else 'no',
            'Source': 'hardness_sr_results.csv',
        })

# ============================================================
# 3-CV comparison: merge 5-fold values from cv_comparison.csv
# ============================================================
# For models that have a 5-fold-evaluable closed form, cv_comparison.py
# computed 5-fold + LOO + LOBO with a consistent KFold seed. Pull those
# values in and overwrite any NaN cells with the live numbers.
cv = maybe_read('cv_comparison.csv')
if cv is not None:
    # Map between cv_comparison Model strings and the unified-table rows
    CV_ALIAS = {
        'Classical HP':                                    ('Classical HP (refit)', None),
        'Power-law HP':                                    ('Power-law HP (refit)', None),
        'M3 (sigma_0(7 elem) + k*d^-1/2)':                 ('M3: σ₀(all elem)', None),
        'PCA-OLS (curated Wen + SD_grain, 6 PCs)':         ('PCA-OLS (curated Wen + SD_grain)', None),
        'SISSO Full (refit)':                              ('SISSO Full', None),
        'SISSO Robust (refit)':                            ('SISSO Robust', None),
        'Compact YS equation (refit)':                       ('Compact YS equation (refit)', 'YS'),
        'Compact HV elbow (refit)':                         ('HV-elbow_F3O2_refit', 'HV'),
    }
    cv_rows_to_add = []
    for _, cv_row in cv.iterrows():
        canonical_target = CV_ALIAS.get(cv_row['Model'])
        if canonical_target is None:
            # Add as new row
            cv_rows_to_add.append({
                'Model': cv_row['Model'],
                'Family': cv_row.get('Family', 'symbolic'),
                'Params': cv_row.get('n_params', np.nan),
                'R2_5fold': cv_row.get('R2_5fold', np.nan),
                'LOO_R2': cv_row.get('R2_LOO', np.nan),
                'LOBO_R2': cv_row.get('R2_LOBO', np.nan),
                'BIC': np.nan,
                'Ext_RMSE_MPa': np.nan,
                'Singularity_safe': 'yes',
                'Source': 'cv_comparison.csv',
            })
            continue
        canon_name, _target = canonical_target
        # Update the matching row if it exists
        matched = False
        for row in rows:
            if str(row['Model']) == canon_name:
                # Always set 5-fold from cv_comparison (cv is the source of truth)
                row['R2_5fold'] = cv_row.get('R2_5fold', np.nan)
                # Backfill LOO/LOBO only if they're NaN (don't clobber)
                if pd.isna(row.get('LOO_R2')):
                    row['LOO_R2'] = cv_row.get('R2_LOO', np.nan)
                if pd.isna(row.get('LOBO_R2')):
                    row['LOBO_R2'] = cv_row.get('R2_LOBO', np.nan)
                matched = True
                break
        if not matched:
            # No existing row — add a new one
            cv_rows_to_add.append({
                'Model': canon_name,
                'Family': cv_row.get('Family', 'symbolic'),
                'Params': cv_row.get('n_params', np.nan),
                'R2_5fold': cv_row.get('R2_5fold', np.nan),
                'LOO_R2': cv_row.get('R2_LOO', np.nan),
                'LOBO_R2': cv_row.get('R2_LOBO', np.nan),
                'BIC': np.nan,
                'Ext_RMSE_MPa': np.nan,
                'Singularity_safe': 'yes',
                'Source': 'cv_comparison.csv',
            })
    rows.extend(cv_rows_to_add)

# ============================================================
# Singularity overrides
# ============================================================
audit = maybe_read('singularity_audit.csv')
# (audit table is informational; the per-equation 'Singularity_safe' is set
# at row creation time based on whether denominators have in-distribution
# near-zeros.)

# ============================================================
# Final table
# ============================================================
if not rows:
    print("[warn] no input CSVs found — table will be empty. Run the analysis "
          "scripts first.")
df_out = pd.DataFrame(rows)
# Enforce column order so 5-fold | LOO | LOBO appear left-to-right
COL_ORDER = ['Model', 'Family', 'Params',
             'R2_5fold', 'LOO_R2', 'LOBO_R2',
             'BIC', 'Ext_RMSE_MPa', 'Singularity_safe', 'Source']
df_out = df_out[[c for c in COL_ORDER if c in df_out.columns]
                + [c for c in df_out.columns if c not in COL_ORDER]]
# Sort by family, then by LOBO (the more honest within-batch-aware metric),
# then by 5-fold; LOO is the tie-breaker
df_out = df_out.sort_values(by=['Family', 'LOBO_R2', 'R2_5fold', 'LOO_R2'],
                            ascending=[True, False, False, False])

out_csv = R / 'unified_model_table.csv'
df_out.to_csv(out_csv, index=False)
print(f"\nUnified table → {out_csv}  ({len(df_out)} rows)")
if len(df_out):
    print("\nPreview (top 20 by LOBO R²):")
    print(df_out.sort_values('LOBO_R2', ascending=False)
          .head(20).to_string(index=False))

# ============================================================
# PNG of the table for inclusion in talks/paper
# ============================================================
if len(df_out):
    fig, ax = plt.subplots(figsize=(13, max(3, 0.35 * len(df_out))))
    ax.axis('off')
    display = df_out.copy().round(3).fillna('—')
    tbl = ax.table(cellText=display.values, colLabels=display.columns,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    ax.set_title('Unified model-selection table (compact-equation + baseline integration)')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/83_unified_model_table.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Wrote {PLOTS_DIR}/83_unified_model_table.png")
