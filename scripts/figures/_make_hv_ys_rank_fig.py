#!/usr/bin/env python3
"""Single-panel HV–YS rank scatter for the Simpson's paradox subsection."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
import _figstyle as S
S.apply()
BASE = str(REPO_ROOT)
OUT  = f'{PAPER_FIG_DIR}/fig_hv_ys_rank.png'

BATCH_COLORS = {b: c for b, (c, _) in S.BATCH.items()}
MARKERS = {b: m for b, (_, m) in S.BATCH.items()}

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df = df.dropna(subset=['HV', 'YS']).copy()
n = len(df)
df['rank_HV'] = df['HV'].rank(ascending=False)
df['rank_YS'] = df['YS'].rank(ascending=False)

rho_g, _ = stats.spearmanr(df['HV'], df['YS'])
batch_rhos = {}
for b in sorted(df['Iteration'].unique()):
    m = df['Iteration'] == b
    if m.sum() >= 5:
        rho_b, _ = stats.spearmanr(df.loc[m, 'HV'], df.loc[m, 'YS'])
        batch_rhos[b] = (rho_b, int(m.sum()))

# Within-campaign (B-campaign = identical processing; C-campaign = swept)
b_mask = df['Iteration'].str.startswith('B')
c_mask = df['Iteration'].str.startswith('C')
rho_B, _ = stats.spearmanr(df.loc[b_mask, 'HV'], df.loc[b_mask, 'YS'])
rho_C, _ = stats.spearmanr(df.loc[c_mask, 'HV'], df.loc[c_mask, 'YS'])
print(f"  B-campaign (n={b_mask.sum()}): rho = {rho_B:+.3f}")
print(f"  C-campaign (n={c_mask.sum()}): rho = {rho_C:+.3f}")

print(f"n = {n}; global rho = {rho_g:.3f}")
for b, (r, k) in batch_rhos.items():
    print(f"  {b}: rho = {r:+.3f} (n={k})")


fig, ax = plt.subplots(figsize=(S.W_COL, 3.2))
lims = [0, n + 1]
ax.plot(lims, lims, 'k--', linewidth=1.2, alpha=0.55, zorder=1, label='perfect agreement')

for b in sorted(df['Iteration'].unique()):
    m = df['Iteration'] == b
    if not m.any():
        continue
    rho_b = batch_rhos.get(b, (np.nan, m.sum()))[0]
    rho_str = f", $\\rho={rho_b:+.2f}$" if not np.isnan(rho_b) else ''
    ax.scatter(df.loc[m, 'rank_YS'], df.loc[m, 'rank_HV'],
               c=BATCH_COLORS.get(b, 'gray'),
               s=55, alpha=0.78, edgecolor='black', linewidth=0.4,
               zorder=3, label=f'{b} ($n={m.sum()}${rho_str})')

ax.set_xlim(lims); ax.set_ylim(lims)
ax.invert_xaxis(); ax.invert_yaxis()
ax.set_xlabel('YS rank (1 = strongest)')
ax.set_ylabel('HV rank (1 = hardest)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', framealpha=0.92, ncol=1)
ax.text(0.04, 0.96,
        (f'Global $\\rho = {rho_g:.2f}$\n'
         f'B-campaign $\\rho = {rho_B:.2f}$\n'
         f'C-campaign $\\rho = {rho_C:.2f}$\n'
         f'Within-batch $\\rho = 0.70\\text{{--}}0.95$'),
        transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='black', linewidth=0.6))
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"Saved {OUT}")
