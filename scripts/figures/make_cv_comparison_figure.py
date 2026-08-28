#!/usr/bin/env python3
"""Produce a grouped-bar comparison of 5-fold vs LOO vs LOBO R² for the
headline models, plus a provisional-5-fold vs LOO/LOBO delta plot."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import RESULTS_DIR, PLOTS_DIR
import _figstyle as S
S.apply()

cv = pd.read_csv(Path(RESULTS_DIR) / 'cv_comparison.csv')
cv['short'] = cv['Model'].str.replace(' (refit)', '', regex=False)
cv['short'] = cv['short'].str.replace('sigma_0', 'σ₀', regex=False)
cv['short'] = cv['short'].str.replace('curated Wen + SD_grain, 6 PCs', 'Wen+SD', regex=False)

# Sort by 5-fold descending for the bar plot
cv_sorted = cv.sort_values('R2_5fold', ascending=False).reset_index(drop=True)

# ============================================================
# 1. 3-CV grouped bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(S.W_FULL, 3.4))
x = np.arange(len(cv_sorted))
w = 0.27

bars5f = ax.bar(x - w, cv_sorted['R2_5fold'], w, label='5-fold CV (random shuffle)',
                color=S.LINEAR, edgecolor='black')
barsLOO = ax.bar(x, cv_sorted['R2_LOO'], w, label='LOO (n=93)',
                 color=S.NONLINEAR, edgecolor='black')
barsLOBO = ax.bar(x + w, cv_sorted['R2_LOBO'], w, label='LOBO (cross-batch)',
                  color=S.OKABE['green'], edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(cv_sorted['short'], rotation=20, ha='right')
ax.set_ylabel('R²')
ax.set_title('Three-way CV comparison: 5-fold vs LOO vs LOBO (cross-batch)',
             fontsize=12)
ax.axhline(0, color='black', lw=0.5)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-0.1, 1.0)
for bars in (bars5f, barsLOO, barsLOBO):
    for b in bars:
        v = b.get_height()
        if not np.isnan(v):
            ax.annotate(f'{v:.2f}', (b.get_x() + b.get_width() / 2, v),
                        ha='center', va='bottom')
plt.savefig(Path(PLOTS_DIR) / '89_cv_comparison_bars.png', dpi=150,
            bbox_inches='tight')
plt.close()

# ============================================================
# 2. Provisional-5-fold vs LOO/LOBO delta plot for the two compact equations
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

# Deck-reported 5-fold values
deck_values = {
    'PCA-OLS (Wen+SD)': 0.675,
    'YS compact (PySR elbow)': 0.811,
    'HV elbow (PySR elbow)': 0.669,
}

# Find matching integration rows
integration = {
    'PCA-OLS (Wen+SD)':
        cv[cv['Model'].str.startswith('PCA-OLS')].iloc[0]
        if not cv[cv['Model'].str.startswith('PCA-OLS')].empty else None,
    'YS compact (PySR elbow)':
        cv[cv['Model'].str.contains('Compact YS', case=False, na=False)].iloc[0]
        if cv['Model'].str.contains('Compact YS', case=False, na=False).any() else None,
    'HV elbow (PySR elbow)':
        cv[cv['Model'].str.contains('HV elbow', case=False, na=False)].iloc[0]
        if cv['Model'].str.contains('HV elbow', case=False, na=False).any() else None,
}

names = list(deck_values.keys())
x = np.arange(len(names))
w = 0.20
deck = [deck_values[n] for n in names]
int_5f = [integration[n]['R2_5fold'] if integration[n] is not None else np.nan for n in names]
int_loo = [integration[n]['R2_LOO'] if integration[n] is not None else np.nan for n in names]
int_lobo = [integration[n]['R2_LOBO'] if integration[n] is not None else np.nan for n in names]

ax.bar(x - 1.5 * w, deck, w, label='provisional 5-fold (reported)',
       color='#888', edgecolor='black')
ax.bar(x - 0.5 * w, int_5f, w, label='Integration 5-fold (live refit)',
       color=S.LINEAR, edgecolor='black')
ax.bar(x + 0.5 * w, int_loo, w, label='Integration LOO',
       color=S.NONLINEAR, edgecolor='black')
ax.bar(x + 1.5 * w, int_lobo, w, label='Integration LOBO',
       color=S.OKABE['green'], edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('R²')
ax.set_title('Provisional 5-fold values vs integrated LOO/LOBO re-evaluation',
             fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='lower left')
ax.set_ylim(0, 1.0)

# Annotate gaps
for i in range(len(names)):
    if not np.isnan(int_lobo[i]):
        gap = deck[i] - int_lobo[i]
        ax.annotate(f'gap {gap:+.2f}',
                    (i, max(deck[i], int_5f[i]) + 0.03),
                    ha='center', color='red')
plt.savefig(Path(PLOTS_DIR) / '90_provisional_vs_integrated.png', dpi=150,
            bbox_inches='tight')
plt.close()

print('Wrote 89_cv_comparison_bars.png and 90_provisional_vs_integrated.png')
