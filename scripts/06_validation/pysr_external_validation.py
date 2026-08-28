#!/usr/bin/env python3
"""
PySR external validation + BIC — judged on the SAME footing as SISSO.
=====================================================================

Closes the gap where PySR equations were listed in the unified table with
BIC = TBD and Ext_RMSE = TBD. This script:

  1. BIC for EVERY PySR grid equation, using the identical formula SISSO used
     (scripts/sisso_analysis.py: compute_ic ->  n*ln(RSS/n) + k*ln(n),
      RSS from the full-data fit, k = number of fitted constants).
  2. External-set validation on the SAME 82-point external dataset SISSO used
     (scripts/external_validation.py: Citrine + Schneider + Otto + Huang,
      training overlaps removed). YS equations scored on YS_exp (n=82);
      HV equations scored on HV_exp (n=25, Huang).
  3. Singularity audit: flags any equation producing non-finite or extreme
     ( |pred| > 1e5 ) predictions on the external set.

ASYMMETRY THIS EXPOSES (reported, not hidden):
  SISSO Full/Robust use ONLY composition + grain size, so they need no
  imputation externally. EVERY PySR feature set uses SD_GS, and F2/F3 also
  use processing (ColdWork, RecrystT, HoldTime) — none of which the external
  literature reports. We therefore impute:
    - SD_GS   <- linear fit SD_GS ~ GrainSize on training (r = 0.80)
    - ColdWork, RecrystT, HoldTime <- training medians
  Each row is flagged with the imputed features it depends on. PySR's external
  numbers are thus conditional on imputation; SISSO's are not. That dependence
  is itself a deployability finding.

Output: results/pysr_external_validation.csv
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR
import external_validation as ev   # guarded by __main__, safe to import

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# ============================================================
# 1. TRAINING DATA + imputation models for missing external features
# ============================================================
print('=' * 72)
print('PySR EXTERNAL VALIDATION + BIC  (same dataset & formulae as SISSO)')
print('=' * 72)

train = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
train = train.loc[:, ~train.columns.duplicated()].copy()
train_ys = train.dropna(subset=['YS']).reset_index(drop=True)
train_hv = train.dropna(subset=['HV']).reset_index(drop=True)

# SD_GS imputation model: SD_GS ~ GrainSize (training)
_sd_fit = LinearRegression().fit(train[['GrainSize']].values, train['SD_GS'].values)
PROC_MEDIANS = {c: float(train[c].median()) for c in ['ColdWork', 'RecrystT', 'HoldTime']}
print(f'SD_GS imputation: SD_GS = {_sd_fit.coef_[0]:.4f}*GrainSize + {_sd_fit.intercept_:.3f} '
      f'(train r={np.corrcoef(train["GrainSize"], train["SD_GS"])[0,1]:.3f})')
print(f'Processing medians (imputed externally): {PROC_MEDIANS}')

# ============================================================
# 2. BUILD EXTERNAL FEATURE TABLE (same 82-point set as SISSO)
# ============================================================
ext = ev.load_all_external_data(train_ys)

def build_feature_frame(df, impute_sd, impute_proc):
    """Compute every feature a PySR equation might reference, per row."""
    rows = []
    for _, r in df.iterrows():
        fracs = {el: float(r.get(f'{el}_frac', 0.0) or 0.0) for el in ELEMENTS}
        d = float(r['GrainSize'])
        desc = ev.compute_hea_descriptors(fracs)
        feat = {f'{el}_frac': fracs[el] for el in ELEMENTS}
        feat.update({
            'd_inv_sqrt': d ** -0.5,
            'GrainSize': d,
            'VEC': desc['VEC'], 'dH_mix': desc['dH_mix'], 'dS_mix': desc['dS_mix'],
            'Omega': min(desc['Omega'], 100.0), 'delta_chi': desc['delta_chi'], 'delta': desc['delta'],
            'mu_bar': desc['mu_bar'], 'delta_mu': desc['delta_mu'], 'Tm_bar': desc['Tm_bar'],
            'Phi_VLC': desc['Phi_VLC'], 'sigma_TC': desc['sigma_TC'], 'a_bar': desc['a_bar'],
        })
        feat['SD_GS'] = float(_sd_fit.predict([[d]])[0]) if impute_sd else np.nan
        for c in ['ColdWork', 'RecrystT', 'HoldTime']:
            feat[c] = PROC_MEDIANS[c] if impute_proc else np.nan
        rows.append(feat)
    return pd.DataFrame(rows, index=df.index)

ext_feat_full = build_feature_frame(ext, impute_sd=True, impute_proc=True)
print(f'\nExternal feature frame: {ext_feat_full.shape[0]} rows x {ext_feat_full.shape[1]} cols')

# ============================================================
# 3. SAFE EVALUATOR for PySR equation strings
# ============================================================
SAFE_FUNCS = {
    'square': lambda x: np.power(x, 2.0),
    'cube':   lambda x: np.power(x, 3.0),
    'sqrt':   lambda x: np.sqrt(np.abs(x)),     # PySR sqrt guards via abs internally
    'log':    lambda x: np.log(np.abs(x) + 1e-12),
    'abs':    np.abs, 'exp': np.exp,
    'inv':    lambda x: 1.0 / x, 'neg': lambda x: -x,
}

# feature sets that each PySR config used (which imputations it depends on)
FS_FEATURES = {
    'F1_grain': ['d_inv_sqrt', 'SD_GS'],
    'F2_full':  [f'{e}_frac' for e in ELEMENTS] + ['ColdWork', 'RecrystT', 'HoldTime', 'd_inv_sqrt', 'SD_GS'],
    'F3_wen':   ['VEC', 'dH_mix', 'dS_mix', 'Omega', 'delta_chi', 'delta',
                 'ColdWork', 'RecrystT', 'HoldTime', 'd_inv_sqrt', 'SD_GS'],
}

def evaluate_equation(eq_str, feat_df):
    """Evaluate a PySR equation string on a feature DataFrame -> np.array preds."""
    ns = {c: feat_df[c].values.astype(float) for c in feat_df.columns}
    ns.update(SAFE_FUNCS)
    preds = eval(eq_str, {'__builtins__': {}}, ns)
    preds = np.asarray(preds, dtype=float)
    if preds.ndim == 0:
        preds = np.full(len(feat_df), float(preds))
    return preds

def compute_bic(fit_r2, y_train, k):
    """SISSO-identical BIC: n*ln(RSS/n) + k*ln(n), RSS from full-data fit R^2."""
    n = len(y_train)
    tss = float(np.sum((y_train - y_train.mean()) ** 2))
    rss = max((1.0 - fit_r2) * tss, 1e-9)
    return n * np.log(rss / n) + k * np.log(n)

# ============================================================
# 4. SCORE EVERY PySR GRID EQUATION
# ============================================================
SISSO_REF = {  # for side-by-side
    'SISSO Full':   dict(BIC=713.99, ExtRMSE=420.61, safe=False),
    'SISSO Robust': dict(BIC=716.57, ExtRMSE=162.93, safe=True),
}

out = []
for target in ['YS', 'HV']:
    grid = pd.read_csv(f'{RESULTS_DIR}/pysr_grid_summary_{target}.csv')
    y_train = (train_ys if target == 'YS' else train_hv)[target].values.astype(float)
    # external truth column
    truth_col = 'YS_exp' if target == 'YS' else 'HV_exp'
    ext_mask = ext[truth_col].notna().values
    y_ext = ext.loc[ext_mask, truth_col].values.astype(float)
    feat_ext = ext_feat_full.loc[ext_mask].reset_index(drop=True)

    # physical sanity ceiling for "singularity-safe": 3x the training max
    extreme_thresh = 3.0 * float(np.max(np.abs(y_train)))

    print(f'\n{"="*72}\n{target}: scoring {len(grid)} PySR equations '
          f'(BIC on n={len(y_train)} train; external on n={len(y_ext)}; '
          f'extreme if |pred|>{extreme_thresh:.0f})\n{"="*72}')

    for _, row in grid.iterrows():
        fs = row['feature_set']; k = int(row['n_constants'])
        eq = str(row['equation']).strip()
        bic = compute_bic(float(row['fit_R2']), y_train, k)
        imputed = [f for f in ['SD_GS', 'ColdWork', 'RecrystT', 'HoldTime'] if f in FS_FEATURES.get(fs, [])]

        ext_r2 = ext_rmse = np.nan
        n_nonfinite = 0; safe = True; note = ''
        if eq and eq.lower() != 'nan':
            try:
                preds = evaluate_equation(eq, feat_ext)
                finite = np.isfinite(preds)
                n_nonfinite = int((~finite).sum())
                extreme = finite & (np.abs(preds) > extreme_thresh)
                safe = (n_nonfinite == 0) and (int(extreme.sum()) == 0)
                # RMSE/R2 on ALL finite points (incl. blow-ups) — matches SISSO's
                # treatment, where the singularity inflated Ext_RMSE to 421.
                if finite.sum() >= 3:
                    ext_r2 = r2_score(y_ext[finite], preds[finite])
                    ext_rmse = float(np.sqrt(mean_squared_error(y_ext[finite], preds[finite])))
                if not safe:
                    note = f'{n_nonfinite} non-finite, {int(extreme.sum())} extreme preds (singularity)'
            except Exception as e:
                note = f'eval failed: {e}'
                safe = False
        else:
            note = 'no equation string (elbow row may be blank)'

        out.append({
            'Model': f"PySR-{target}-{fs}_{row['op_set']}_{row['selection']}",
            'Target': target, 'feature_set': fs, 'op_set': row['op_set'],
            'selection': row['selection'], 'Params': k, 'complexity': int(row['complexity']),
            'fit_R2': round(float(row['fit_R2']), 4),
            'LOO_R2': round(float(row['cv_refit_LOO_R2']), 4),
            'LOBO_R2': round(float(row['cv_refit_LOBO_R2']), 4),
            'BIC': round(bic, 1),
            'Ext_R2': round(ext_r2, 4) if np.isfinite(ext_r2) else '',
            'Ext_RMSE_MPa': round(ext_rmse, 1) if np.isfinite(ext_rmse) else '',
            'n_ext': len(y_ext),
            'Singularity_safe': 'yes' if safe else 'no',
            'imputed_features': '+'.join(imputed) if imputed else 'none',
            'note': note,
        })

res = pd.DataFrame(out)
res.to_csv(f'{RESULTS_DIR}/pysr_external_validation.csv', index=False)
print(f'\nWrote results/pysr_external_validation.csv ({len(res)} equations scored)')

# ============================================================
# 5. SUMMARY — best PySR vs SISSO on BIC + external RMSE
# ============================================================
for target in ['YS', 'HV']:
    sub = res[(res.Target == target) & (res.Ext_RMSE_MPa != '')].copy()
    if sub.empty:
        continue
    sub['Ext_RMSE_MPa'] = sub['Ext_RMSE_MPa'].astype(float)
    sub['BIC'] = sub['BIC'].astype(float)
    print(f'\n{"="*72}\n{target}: PySR judged on equal footing (sorted by Ext_RMSE)\n{"="*72}')
    show = sub.sort_values('Ext_RMSE_MPa').head(6)
    print(show[['Model', 'Params', 'fit_R2', 'LOO_R2', 'BIC',
                'Ext_R2', 'Ext_RMSE_MPa', 'Singularity_safe', 'imputed_features']].to_string(index=False))
    best_bic = sub.sort_values('BIC').head(3)
    print(f'\n  Best PySR-{target} by BIC:')
    print(best_bic[['Model', 'Params', 'BIC', 'Ext_RMSE_MPa', 'Singularity_safe']].to_string(index=False))

print(f'\n{"="*72}\nSISSO reference (composition+grain only, NO imputation needed):')
for k, v in SISSO_REF.items():
    print(f'  {k:14s}  BIC={v["BIC"]:.1f}  Ext_RMSE={v["ExtRMSE"]:.1f} MPa  safe={v["safe"]}')
print('='*72)
print('\nDone.')
