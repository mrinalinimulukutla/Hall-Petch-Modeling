#!/usr/bin/env python3
"""Family 4 matched-input LOBO heatmap (main2 Fig. 7 / paper Fig. fair_heatmap).

Reads results/fair_comparison.csv (no model re-run) and renders cross-cluster
LOBO R^2 for every model x feature-set cell, for YS and HV side by side.

Design notes (2026-08 revision):
  * Diverging Okabe-Ito palette centred on R^2 = 0, the meaningful reference
    ("no better than predicting the held-out mean").  The previous version
    used viridis clipped at vmin=0, which collapsed 48 of 51 HV cells onto a
    single colour and hid the whole HV result.
  * Models are grouped by family (linear block, then non-linear block) with a
    divider, so family is encoded by POSITION as well as colour -- readable
    without colour vision.
  * One shared row order across both panels so YS and HV are comparable
    row-by-row.
  * Large fonts sized for a two-column \\textwidth (figure*) placement.

Output: analysis_plots/fair_comparison_LOBO_heatmap.png (+ copies to the
paper figure directories).
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _config import RESULTS_DIR, PLOTS_DIR, PAPER_FIG_DIR, REPO_ROOT
import _figstyle as S
S.apply()

# ---------------------------------------------------------------- palette ---
# Okabe-Ito vermillion (negative) -> white (zero) -> blue (positive).
# Both hues stay distinguishable under deuteranopia and protanopia.
CMAP = LinearSegmentedColormap.from_list(
    'okabe_div', ['#7A2E00', '#D55E00', '#F0A868', '#FFFFFF',
                  '#8FC1E3', '#0072B2', '#00436B'])
VMIN, VMAX = -1.0, 0.7          # colour range; true values still printed
NORM = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)

SET_LABEL = {
    'S1_grain':    'S1\n$d^{-1/2}$, SD',
    'S2_wen':      'S2\n+Wen',
    'S3_wen_proc': 'S3\n+proc.',
    'S4_phys':     'S4\n+comp.',
}
LADDER = ['S1_grain', 'S2_wen', 'S3_wen_proc', 'S4_phys']

FS_TICK, FS_CELL, FS_TITLE, FS_LAB = S.FS_TICK, 7.6, S.FS_TITLE, S.FS_ANNOT

df = pd.read_csv(f'{RESULTS_DIR}/fair_comparison.csv')
family = df.drop_duplicates('Model').set_index('Model')['Family'].to_dict()

# shared row order: linear block first, each block sorted by best YS LOBO
best_ys = df[df.Target == 'YS'].groupby('Model')['R2_LOBO'].max()
lin = sorted([m for m in family if family[m] == 'linear'],
             key=lambda m: -best_ys.get(m, -9))
non = sorted([m for m in family if family[m] != 'linear'],
             key=lambda m: -best_ys.get(m, -9))
ORDER = lin + non
SPLIT = len(lin)                      # divider sits after the linear block

fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 4.3))
for ax, tgt in zip(axes, ['YS', 'HV']):
    piv = (df[df.Target == tgt]
           .pivot_table(index='Model', columns='FeatureSet', values='R2_LOBO')
           .reindex(index=ORDER, columns=LADDER))
    V = piv.values.astype(float)

    im = ax.imshow(np.clip(V, VMIN, VMAX), cmap=CMAP, norm=NORM, aspect='auto')

    ax.set_xticks(range(V.shape[1]))
    ax.set_xticklabels([SET_LABEL[c] for c in piv.columns], fontsize=7.8)
    ax.set_yticks(range(V.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8.2)
    for i, m in enumerate(piv.index):
        ax.get_yticklabels()[i].set_color(
            '#0072B2' if family[m] == 'linear' else '#B04000')
    ax.tick_params(length=0)

    # cell values; colour chosen for contrast against the local background
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            if not np.isfinite(v):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor='#E8E8E8',
                                       edgecolor='white', linewidth=1.6, zorder=2))
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=FS_CELL,
                        color='#777777', style='italic', zorder=3)
                continue
            shade = NORM(np.clip(v, VMIN, VMAX))
            txt = 'white' if (shade < 0.18 or shade > 0.88) else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=FS_CELL, color=txt,
                    fontweight='bold' if v > 0 else 'normal')
            if v > 0 and tgt == 'HV':       # rare positives are worth flagging
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor='#111111', linewidth=1.3))

    # no grid: the cells carry the structure on their own
    ax.grid(False)
    ax.axhline(SPLIT - 0.5, color='#111111', linewidth=1.6)
    if tgt == 'YS':                       # group labels once, in the left margin
        n = V.shape[0]
        ax.text(-0.40, 1 - (SPLIT / 2) / n, 'LINEAR', rotation=90,
                va='center', ha='center', fontsize=S.FS_ANNOT, fontweight='bold',
                color='#0072B2', transform=ax.transAxes, clip_on=False)
        ax.text(-0.40, 1 - ((SPLIT + n) / 2) / n, 'NON-LINEAR', rotation=90,
                va='center', ha='center', fontsize=S.FS_ANNOT, fontweight='bold',
                color='#B04000', transform=ax.transAxes, clip_on=False)

    S.title(ax, f'{tgt}: cross-cluster (LOBO) $R^2$', size=S.FS_ANNOT + 1.5, loc='center')

cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.025,
                  ticks=[-1.0, -0.5, 0.0, 0.25, 0.5, 0.7])
cb.set_label('LOBO $R^2$  (0 = held-out mean)', fontsize=FS_LAB)
cb.ax.tick_params(labelsize=7.8)
cb.ax.set_yticklabels(['$\\leq-1.0$', '-0.5', '0.0', '0.25', '0.50', '0.70'])


out = f'{PLOTS_DIR}/fair_comparison_LOBO_heatmap.png'
fig.savefig(out)
print('Wrote', out)

for dest in (str(PAPER_FIG_DIR),):
    if os.path.isdir(dest):
        shutil.copy(out, dest)
        print('  copied ->', dest)
