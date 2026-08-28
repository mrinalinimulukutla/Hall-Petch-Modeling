#!/usr/bin/env python3
"""
PySR Outer-Loop Cross-Validation (gold-standard generalization estimate)
=========================================================================
For each LOBO fold:
  1. Hold out one batch.
  2. Re-run PySR from scratch on the remaining training data using the
     F3+O3 configuration (Wen descriptors + SD_grain + processing,
     +, -, *, /, sqrt, square, log operators).
  3. Record the elbow-selected equation discovered in that fold.
  4. Apply that fold-specific equation to the held-out batch.

Compare fold-discovered equations to the full-data elbow equation. If the
form changes substantially across folds, the headline equation is fold-
specific rather than universal.

Cost: ~10 min per fold × 6 folds = ~60 min for YS (set niterations=20 to
trim).
"""
import warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import DATA_DIR, RESULTS_DIR, REPO_ROOT

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# ============================================================
# 1. LOAD
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv').dropna(subset=['YS']).reset_index(drop=True)
if 'Iteration' not in df.columns:
    raise SystemExit("[fatal] no 'Iteration' column for LOBO.")
batches = sorted(df['Iteration'].unique())

SD_COL = next((c for c in ('SD_GS', 'GrainSize_SD', 'SD_grain', 'GS_SD') if c in df.columns), None)
if SD_COL is None:
    raise SystemExit("[fatal] no SD_grain column.")

feats = ['VEC', 'dH_mix', 'delta_chi', 'Omega', 'dS_mix', 'delta',
         'd_inv_sqrt', SD_COL, 'ColdWork', 'RecrystT', 'HoldTime']
feats = [c for c in feats if c in df.columns]
print(f"Outer-loop CV: F3+O3, {len(feats)} features, {len(batches)} folds")

X = df[feats].values.astype(float)
y = df['YS'].values.astype(float)
g = df['Iteration'].values

# ============================================================
# 2. PER-FOLD PYSR
# ============================================================
from pysr import PySRRegressor

rows = []
for held in batches:
    t0 = time.time()
    tr = g != held
    te = ~tr
    print(f"\n--- hold-out batch {held}  (n_train={tr.sum()}, n_test={te.sum()}) ---")
    model = PySRRegressor(
        niterations=20,              # ~half of grid run to keep folds tractable
        populations=20,
        population_size=40,
        maxsize=25,
        maxdepth=6,
        parsimony=0.005,
        model_selection='best',
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square", "log"],
        temp_equation_file=True,
        verbosity=0,
        progress=False,
        random_state=42,
        deterministic=True,
        parallelism='serial',
    )
    try:
        model.fit(X[tr], y[tr], variable_names=feats)
        eqs = model.equations_
        idx = int(eqs['score'].idxmax()) if 'score' in eqs else int(eqs['loss'].idxmin())
        equation = str(eqs.iloc[idx]['equation'])
        complexity = int(eqs.iloc[idx]['complexity'])
        y_pred = model.predict(X[te], idx)
        r2_test = r2_score(y[te], y_pred)
        rmse_test = float(np.sqrt(mean_squared_error(y[te], y_pred)))
        rows.append({
            'held_out_batch': held,
            'complexity': complexity,
            'fold_R2_test': r2_test,
            'fold_RMSE_test': rmse_test,
            'discovered_equation': equation,
            'wallclock_s': round(time.time() - t0, 1),
        })
        print(f"  R²_test = {r2_test:6.3f}, RMSE = {rmse_test:6.1f}  "
              f"({time.time() - t0:.0f} s)")
        print(f"  equation: {equation[:120]}")
    except Exception as exc:
        rows.append({
            'held_out_batch': held, 'complexity': -1,
            'fold_R2_test': np.nan, 'fold_RMSE_test': np.nan,
            'discovered_equation': f'ERROR: {exc}',
            'wallclock_s': round(time.time() - t0, 1),
        })
        print(f"  [error] {exc}")

# ============================================================
# 3. SAVE
# ============================================================
out = pd.DataFrame(rows)
out.to_csv(f'{RESULTS_DIR}/pysr_outer_loop_cv.csv', index=False)
print(f"\nWrote {RESULTS_DIR}/pysr_outer_loop_cv.csv")
print(out[['held_out_batch', 'complexity', 'fold_R2_test',
           'fold_RMSE_test', 'wallclock_s']].to_string(index=False))

agg_r2 = out['fold_R2_test'].mean()
agg_rmse = float(np.sqrt((out['fold_RMSE_test'] ** 2).mean()))
print(f"\nAggregate outer-loop R² = {agg_r2:.3f}, RMSE = {agg_rmse:.1f} MPa")
print("Compare to full-data PySR (no held-out batch) for the generalization gap.")
