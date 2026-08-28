#!/usr/bin/env python3
"""Graphical abstract: the four measured failure modes -> guardrails.
Four pastel cards (Physics / Descriptors / Validation / Deployment) in the
rounded-panel TOC style. Every number shown is canonical (CLAUDE.md sec. 3).

Pure matplotlib, reads no data; part of `make figures`.
Writes paper/figures/fig_graphical_abstract.png.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle, RegularPolygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import PAPER_FIG_DIR

EDGE = '#2b2b2b'
PANELS = [
    dict(head='Physics',     bg='#e9f1fa', fg='#16456e'),
    dict(head='Descriptors', bg='#fdf3dc', fg='#a8681c'),
    dict(head='Validation',  bg='#fbe5e2', fg='#a41e1e'),
    dict(head='Deployment',  bg='#e8f2e4', fg='#2d6a2d'),
]

FW, FH = 14.8, 5.1
fig = plt.figure(figsize=(FW, FH), dpi=100)
fig.patch.set_facecolor('white')
fig.text(0.010, 0.980, 'Building and auditing ML strength models for FCC HEAs',
         fontsize=29, fontweight='bold', ha='left', va='top', color='black')

W, H, Y0 = 0.228, 0.62, 0.045
X0S = [0.010, 0.258, 0.506, 0.754]
axes = []
for x0, p in zip(X0S, PANELS):
    fig.text(x0 + W / 2, Y0 + H + 0.032, p['head'], fontsize=23,
             fontweight='bold', ha='center', va='bottom', color='black')
    ax = fig.add_axes([x0, Y0, W, H]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.015, 0.015), 0.97, 0.97,
                                boxstyle='round,pad=0.002,rounding_size=0.06',
                                facecolor=p['bg'], edgecolor=EDGE, linewidth=3.2,
                                mutation_aspect=(H / W) * (FH / FW)))
    axes.append(ax)
for xc in [x + W + 0.010 for x in X0S[:-1]]:
    fig.patches.append(Polygon([[xc - 0.005, 0.41], [xc - 0.005, 0.30], [xc + 0.010, 0.355]],
                               closed=True, facecolor='#1a1a1a', transform=fig.transFigure))

def label(ax, txt, fg):
    ax.text(0.5, 0.905, txt, fontsize=14.8, fontweight='bold', color=fg,
            ha='center', va='top', linespacing=1.3)

# 1 ---- Physics: SSS theories overpredict (parity, points above y = x) ------
ax, fg = axes[0], PANELS[0]['fg']
label(ax, 'SSS theories overpredict\nby 2–5×', fg)
ax.plot([0.14, 0.82], [0.10, 0.52], color='#777', lw=2.2, ls='--')
rng = np.random.default_rng(5)
xs = np.linspace(0.20, 0.70, 6)
ys = 0.12 + (xs - 0.14) * 1.05 + rng.normal(0, 0.025, 6) + 0.10
ax.scatter(xs, np.clip(ys, 0, 0.64), s=64, facecolor='#6da4d8',
           edgecolor=fg, lw=1.7, zorder=5)
ax.text(0.83, 0.47, 'y = x', fontsize=12.5, color='#666', style='italic')

# 2 ---- Descriptors: redundant given composition (nested sets) --------------
ax, fg = axes[1], PANELS[1]['fg']
label(ax, 'redundant given\ncomposition  (r < 0.10)', fg)
ax.add_patch(Circle((0.47, 0.35), 0.275, facecolor='#f6dfae', edgecolor=fg, lw=2.6))
ax.add_patch(Circle((0.56, 0.28), 0.12, facecolor='#e8bd74', edgecolor=fg, lw=2.2))
ax.text(0.40, 0.45, 'composition', fontsize=12.5, color=fg, ha='center',
        fontweight='bold')
ax.text(0.56, 0.28, 'SSS', fontsize=11.5, color='#5c3a10', ha='center',
        va='center', fontweight='bold')

# 3 ---- Validation: LOO flatters, LOBO is honest ------------------------------
ax, fg = axes[2], PANELS[2]['fg']
label(ax, 'LOO flatters:\n0.73 → LOBO 0.57', fg)
bars = [('5-fold', 0.44, '#f0b26b'), ('LOO', 0.44, '#9a9ad0'), ('LOBO', 0.33, '#c62828')]
for i, (lab, hgt, c) in enumerate(bars):
    ax.add_patch(Rectangle((0.15 + i * 0.26, 0.13), 0.17, hgt,
                           facecolor=c, edgecolor=EDGE, lw=1.8))
    ax.text(0.15 + i * 0.26 + 0.085, 0.095, lab, fontsize=12, ha='center',
            va='top', color='#333', fontweight='bold')
ax.annotate('', xy=(0.15 + 2 * 0.26 + 0.085, 0.13 + 0.33 + 0.025),
            xytext=(0.15 + 0.085, 0.13 + 0.44 + 0.025),
            arrowprops=dict(arrowstyle='-|>', color=fg, lw=4.0))

# 4 ---- Deployment: singularity audit + external accuracy audit --------------
ax, fg = axes[3], PANELS[3]['fg']
label(ax, 'singularity + external\naudits: 421 → 163 MPa', fg)
# bounded (deployable) form: passes both audits
x = np.linspace(0.10, 0.90, 200)
ax.plot(x, 0.24 + 0.08 * np.sin(6.0 * (x - 0.1)) + 0.13 * x, color=fg,
        lw=3.4, solid_capstyle='round')
ax.text(0.24, 0.20, 'bounded', fontsize=11.5, color=fg, style='italic')
ax.add_patch(Circle((0.90, 0.42), 0.062, facecolor='#8fca8f', edgecolor=fg, lw=2.2))
ax.text(0.90, 0.42, '\u2713', fontsize=15, ha='center', va='center', color=fg,
        fontweight='bold')
# divergent form: denominator -> 0 sends the prediction to infinity
xs = np.linspace(0.42, 0.655, 100)
ax.plot(xs, np.minimum(0.26 + 0.04 / (0.70 - xs), 0.58), color='#c62828',
        lw=2.6, ls='--')
ax.scatter([0.65], [0.575], s=70, facecolor='#ffd54f', edgecolor='#c62828',
           lw=2.2, zorder=6)
ax.text(0.505, 0.545, '$\\delta_\\mu\\!\\to\\!0$', fontsize=12.5,
        color='#c62828', ha='right', fontweight='bold')
ax.text(0.72, 0.575, '$\\to\\infty$', fontsize=13, color='#c62828',
        ha='left', va='center', fontweight='bold')

out = os.path.join(str(PAPER_FIG_DIR), 'fig_graphical_abstract.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Wrote {out}')
