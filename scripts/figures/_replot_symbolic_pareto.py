#!/usr/bin/env python3
"""fig08 — SR results slide (YS, F3 curated-Wen + O3):
PySR training loss vs equation COMPLEXITY (node count), with the elbow and
accuracy picks marked."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import RESULTS_DIR, PAPER_FIG_DIR

OUT_FIG = f'{PAPER_FIG_DIR}/fig08_symbolic_pareto.png'
front = pd.read_csv(f'{RESULTS_DIR}/pysr_grid/YS__F3_wen__O3_full_pareto.csv').sort_values('complexity')
ELBOW_C, ACC_C = 9, 30

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})
fig, ax = plt.subplots(figsize=(6.4, 4.8))

ax.plot(front['complexity'], front['loss'], 'o-', color='#1f77b4', markersize=5,
        linewidth=1.6, label='PySR front', zorder=2)
def mark(c, label, marker, color):
    row = front[front['complexity'] == c].iloc[0]
    ax.scatter([c], [row['loss']], marker=marker, s=220, color=color,
               edgecolor='black', linewidth=1.0, zorder=5, label=label)
mark(ELBOW_C, 'elbow (2 const)', 'D', '#d62728')
mark(ACC_C,  'accuracy (6 const)', '*', '#2ca02c')
ax.set_xlabel('Equation complexity (node count)')
ax.set_ylabel('Training loss (MSE, MPa$^2$)')
ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper right', framealpha=0.92); ax.set_xlim(left=0)
plt.savefig(OUT_FIG, dpi=200, bbox_inches='tight')
print(f"Saved {OUT_FIG}  (single panel: loss vs complexity)")
