#!/usr/bin/env python3
"""Fig. 1 — framework overview schematic (the staircase + guardrails).

Graphical abstract for the best-practices framework: a five-tier ascent in
model flexibility, each tier with the capability gained and the named risk it
introduces, over a cluster-out validation floor, with the four pillars ->
guardrails panel. Demonstrated on FCC HEAs; protocol is material-class-agnostic.

Output: paper/figures/fig00_framework_overview.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _config import PAPER_FIG_DIR

NAVY = '#1F3A5F'; ACCENT = '#C84B31'; TEAL = '#2A7F7E'; AMBER = '#D98A1F'
LIGHT = '#EAF0F6'; GREEN = '#2A7F2A'; GRAY = '#666666'

fig, ax = plt.subplots(figsize=(13, 6.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis('off')

ax.text(0.2, 6.25, 'A best-practices framework for physics-informed property prediction',
        fontsize=16, weight='bold', color=NAVY)
ax.text(0.2, 5.9, 'Each tier gains predictive capability and incurs a specific, named risk the framework manages',
        fontsize=10.5, style='italic', color=GRAY)

# ----- staircase steps -----
steps = [
    ('Classical\nHall–Petch', 'universal backbone\n(grain size)', 'no added risk', GREEN),
    ('+ Physics\ndescriptors', 'SSS, Wen, PCA', 'descriptor redundancy\n(partial r < 0.10)', ACCENT),
    ('+ Composition /\nprocessing', 'where signal lives\n(σ₀ vs k)', 'interpretability\nvs fit', AMBER),
    ('+ Non-linear ML\n(ARMOTE-CV)', 'captures\ninteractions', 'validation optimism\n(LOO flatters)', ACCENT),
    ('→ Symbolic\nregression', 'interpretable\nclosed-form laws', 'deployment singularity\n(ext 421→163)', ACCENT),
]
x0, w, gap = 0.5, 2.0, 0.35
base_y = 2.15
for i, (title, gain, risk, rc) in enumerate(steps):
    x = x0 + i * (w + gap)
    h = 1.45 + i * 0.42                     # ascending height (min raised to avoid crowding)
    y = base_y
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                         linewidth=1.6, edgecolor=NAVY, facecolor=LIGHT)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='top', fontsize=10.5, weight='bold', color=NAVY)
    ax.text(x + w / 2, y + 0.30, gain, ha='center', va='bottom', fontsize=8.6, color=TEAL, style='italic')
    # rising connector arrow
    if i < len(steps) - 1:
        nx = x0 + (i + 1) * (w + gap)
        nh = 1.0 + (i + 1) * 0.45
        ax.add_patch(FancyArrowPatch((x + w, y + h), (nx, y + nh - 0.05),
                     arrowstyle='-|>', mutation_scale=16, color=NAVY, lw=1.6))
    # risk tag below
    rtag = FancyBboxPatch((x, base_y - 1.05), w, 0.78, boxstyle='round,pad=0.02,rounding_size=0.06',
                          linewidth=0, facecolor=rc, alpha=0.92)
    ax.add_patch(rtag)
    label = ('⚠ ' if rc != GREEN else '✓ ') + risk
    ax.text(x + w / 2, base_y - 0.66, label, ha='center', va='center', fontsize=7.8,
            color='white', weight='bold')

# ----- validation floor -----
fy = 0.55
floor = FancyBboxPatch((0.5, fy), 11.2, 0.55, boxstyle='round,pad=0.02,rounding_size=0.06',
                       linewidth=0, facecolor=NAVY)
ax.add_patch(floor)
ax.text(6.1, fy + 0.275, 'VALIDATION FLOOR   5-fold  →  LOO  →  LOBO (batch-out)  →  external  +  singularity audit',
        ha='center', va='center', fontsize=10, color='white', weight='bold')

# ----- right-hand pillars/guardrails panel -----
px = 11.95
ax.add_patch(FancyBboxPatch((px, base_y - 1.05), 0.95, 4.0,
             boxstyle='round,pad=0.02,rounding_size=0.06', linewidth=1.4,
             edgecolor=TEAL, facecolor='white'))
ax.text(px + 0.48, base_y + 2.78, 'PILLARS', ha='center', fontsize=9, weight='bold', color=TEAL, rotation=0)
for j, p in enumerate(['physics', 'descriptors', 'validation', 'deployment']):
    ax.text(px + 0.48, base_y + 2.35 - j * 0.62, p, ha='center', va='center',
            fontsize=8.6, color=NAVY, weight='bold', rotation=0)

# ----- footer band -----
ax.add_patch(FancyBboxPatch((0.5, 0.04), 12.4, 0.38, boxstyle='square,pad=0',
             linewidth=0, facecolor=AMBER, alpha=0.18))
ax.text(6.7, 0.23, 'Demonstrated on FCC HEAs (YS, HV)  —  the protocol is material-class-agnostic '
        '(BCC HEAs, conventional alloys, ceramics, composites)',
        ha='center', va='center', fontsize=9.5, color='#7a4d00', weight='bold')

out = f'{PAPER_FIG_DIR}/fig00_framework_overview.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('Wrote', out)
