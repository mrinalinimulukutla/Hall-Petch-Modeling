#!/usr/bin/env python3
"""ARCHIVAL PROVENANCE SCRIPT — not runnable from this repository alone.

Builds the baseline-stream (original, no SD_GS) vs SD_GS-augmented
comparison that produced results/sdgrain_model_comparison.csv (the M13/M15
source table). It reads the two original campaign folders, which are not
distributed with this repo; the cached CSV in results/ is the record.

Original docstring: build original (no SD_GS) vs modified (+SD_GS)
comparison artefacts.

For each model in the four core analyses (ML panel, composition-HP
hierarchy, SISSO, external validation), report side-by-side:
    original (no SD_GS)  vs  SD_GS-augmented  =  Δ (with SD_GS)

Output:
    hall_petch_may_2026/results/sdgrain_model_comparison.csv
    hall_petch_may_2026/analysis_plots/91_sdgrain_model_comparison.png
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).resolve().parent.parent
HALL_PETCH = ROOT
ORIG_STREAM_DIR = ROOT.parent / 'baseline_stream_original'      # original campaign folder (not distributed)
MOD_STREAM_DIR  = ROOT.parent / 'baseline_stream_sdgs_augmented'  # SD_GS-augmented rerun folder (not distributed)

OUT_CSV = HALL_PETCH / 'results' / 'sdgrain_model_comparison.csv'
OUT_PNG = HALL_PETCH / 'analysis_plots' / '91_sdgrain_model_comparison.png'


def _read(p):
    return pd.read_csv(p) if p.exists() else None


rows = []

# -------- 1. ML panel: model_search_results.csv --------
orig_ml = _read(ORIG_STREAM_DIR / 'results' / 'model_search_results.csv')
mod_ml  = _read(MOD_STREAM_DIR  / 'results' / 'model_search_results.csv')
if orig_ml is not None and mod_ml is not None:
    # Match on Model name; for the panel we expect identical model lists
    orig_by_name = {r['Model']: r for _, r in orig_ml.iterrows()}
    mod_by_name  = {r['Model']: r for _, r in mod_ml.iterrows()}
    for name in set(orig_by_name) | set(mod_by_name):
        o = orig_by_name.get(name, {})
        m = mod_by_name.get(name, {})
        rows.append({
            'analysis': 'ML panel',
            'Model': name,
            'orig_5fold':  o.get('R2_5fold', np.nan) if isinstance(o, dict) or hasattr(o, 'get') else np.nan,
            'orig_LOO':    o.get('LOO_R2', np.nan)   if hasattr(o, 'get') else np.nan,
            'orig_LOBO':   o.get('LOBO_R2', np.nan)  if hasattr(o, 'get') else np.nan,
            'mod_5fold':   m.get('R2_5fold', np.nan) if hasattr(m, 'get') else np.nan,
            'mod_LOO':     m.get('LOO_R2', np.nan)   if hasattr(m, 'get') else np.nan,
            'mod_LOBO':    m.get('LOBO_R2', np.nan)  if hasattr(m, 'get') else np.nan,
        })

# -------- 2. composition-HP M-models: comp_hp_model_comparison.csv --------
# The baseline stream's CSV uses elpd_loo as the Bayesian metric. We surface both
# the orig (M0–M12) and mod (M0–M15) results.
orig_chp = _read(ORIG_STREAM_DIR / 'results' / 'comp_hp_model_comparison.csv')
mod_chp  = _read(MOD_STREAM_DIR  / 'results' / 'comp_hp_model_comparison.csv')
if orig_chp is not None and mod_chp is not None:
    # First column is the model name; format may include 'name' or unnamed
    namecol_o = orig_chp.columns[0]
    namecol_m = mod_chp.columns[0]
    orig_by = {r[namecol_o]: r for _, r in orig_chp.iterrows()}
    mod_by  = {r[namecol_m]: r for _, r in mod_chp.iterrows()}
    for name in set(orig_by) | set(mod_by):
        o = orig_by.get(name, {})
        m = mod_by.get(name, {})
        rows.append({
            'analysis': 'comp-HP hierarchy',
            'Model': name,
            'orig_5fold':  np.nan,
            'orig_LOO':    o.get('elpd_loo', np.nan) if hasattr(o, 'get') else np.nan,
            'orig_LOBO':   np.nan,
            'mod_5fold':   np.nan,
            'mod_LOO':     m.get('elpd_loo', np.nan) if hasattr(m, 'get') else np.nan,
            'mod_LOBO':    np.nan,
        })

# -------- 3. SISSO: sisso_results.csv --------
orig_s = _read(ORIG_STREAM_DIR / 'results' / 'sisso_results.csv')
mod_s  = _read(MOD_STREAM_DIR  / 'results' / 'sisso_results.csv')
if orig_s is not None and mod_s is not None:
    orig_by = {r['Model']: r for _, r in orig_s.iterrows()}
    mod_by  = {r['Model']: r for _, r in mod_s.iterrows()}
    for name in set(orig_by) | set(mod_by):
        o = orig_by.get(name, {})
        m = mod_by.get(name, {})
        rows.append({
            'analysis': 'SISSO',
            'Model': name,
            'orig_5fold':  np.nan,
            'orig_LOO':    o.get('LOO_R2', np.nan)  if hasattr(o, 'get') else np.nan,
            'orig_LOBO':   o.get('LOBO_R2', np.nan) if hasattr(o, 'get') else np.nan,
            'mod_5fold':   np.nan,
            'mod_LOO':     m.get('LOO_R2', np.nan)  if hasattr(m, 'get') else np.nan,
            'mod_LOBO':    m.get('LOBO_R2', np.nan) if hasattr(m, 'get') else np.nan,
        })

# Build dataframe + Δ columns
df = pd.DataFrame(rows)
for prot in ('5fold', 'LOO', 'LOBO'):
    df[f'd_{prot}'] = df[f'mod_{prot}'] - df[f'orig_{prot}']

# Save
df.to_csv(OUT_CSV, index=False)
print(f'Wrote {OUT_CSV}  ({len(df)} rows)')
print()
print('=== top |Δ LOO| ===')
print(df.dropna(subset=['d_LOO']).reindex(df['d_LOO'].abs().sort_values(ascending=False).index)[
    ['analysis', 'Model', 'orig_LOO', 'mod_LOO', 'd_LOO']].head(15).to_string(index=False))
print()
print('=== top |Δ LOBO| ===')
print(df.dropna(subset=['d_LOBO']).reindex(df['d_LOBO'].abs().sort_values(ascending=False).index)[
    ['analysis', 'Model', 'orig_LOBO', 'mod_LOBO', 'd_LOBO']].head(15).to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ml_only = df[df['analysis'] == 'ML panel'].dropna(subset=['orig_LOBO', 'mod_LOBO'])
if len(ml_only):
    x = np.arange(len(ml_only))
    axes[0].barh(x - 0.2, ml_only['orig_LOBO'], 0.4, label='Original (no SD_GS)', color='#888')
    axes[0].barh(x + 0.2, ml_only['mod_LOBO'], 0.4, label='+ SD_GS (mod)', color='#4c72b0')
    axes[0].set_yticks(x); axes[0].set_yticklabels(ml_only['Model'], fontsize=8)
    axes[0].set_xlabel('LOBO R²')
    axes[0].set_title('ML panel: original vs SD_GS-augmented')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)

ss = df[df['analysis'] == 'SISSO'].dropna(subset=['orig_LOO', 'mod_LOO'])
if len(ss):
    x = np.arange(len(ss))
    axes[1].barh(x - 0.2, ss['orig_LOO'], 0.4, label='Orig LOO', color='#888')
    axes[1].barh(x + 0.2, ss['mod_LOO'], 0.4, label='+SD_GS LOO', color='#4c72b0')
    axes[1].barh(x - 0.2, ss['orig_LOBO'], 0.4, alpha=0.4)
    axes[1].barh(x + 0.2, ss['mod_LOBO'], 0.4, alpha=0.4)
    axes[1].set_yticks(x); axes[1].set_yticklabels(ss['Model'], fontsize=9)
    axes[1].set_xlabel('R²')
    axes[1].set_title('SISSO: original vs SD_GS-augmented')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nWrote {OUT_PNG}')
