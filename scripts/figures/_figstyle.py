#!/usr/bin/env python3
"""Single source of truth for figure colour and typography.

Every figure in the paper and the SI imports this module, so the whole
document uses one palette and one set of font sizes.  The only deliberate
exception is `fig00_framework_overview.png`, which keeps its own scheme.

Palette
-------
Okabe--Ito (Okabe & Ito 2008), the standard eight-colour set chosen so that
no two entries collide under deuteranopia, protanopia or tritanopia.  Colour
is never the only channel: batches also carry distinct marker shapes, model
families are separated by position, and signed quantities print their value.

Colour maps
-----------
DIVERGING   signed quantities (R^2, correlations, coefficients).  Centred on
            zero: vermillion below, blue above, white at zero.
SEQUENTIAL  magnitudes (|r|, hull overlap, counts).  Single blue hue, so it
            is monotonic in lightness and safe in greyscale.

Usage
-----
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _figstyle import apply, OKABE, BATCH, DIVERGING, SEQUENTIAL, save
    apply()
"""
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ colours --
OKABE = {
    'black':     '#000000',
    'orange':    '#E69F00',
    'skyblue':   '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
    'blue':      '#0072B2',
    'vermilion': '#D55E00',
    'purple':    '#CC79A7',
}

# ordered cycle for generic categorical series
CYCLE = [OKABE['blue'], OKABE['vermilion'], OKABE['green'], OKABE['purple'],
         OKABE['orange'], OKABE['skyblue'], '#7A2E00', OKABE['black']]

# semantic accents used across the paper
LINEAR      = OKABE['blue']        # linear / interpretable models
NONLINEAR   = '#B04000'            # non-linear / black-box models
REFERENCE   = '#4D4D4D'            # baselines, 1:1 lines, guides
HIGHLIGHT   = OKABE['orange']      # the one thing the reader should look at
GRIDCOLOR   = '#D9D9D9'

# experimental batches: colour AND marker, so the encoding is redundant
BATCH = {
    'BBA': (OKABE['vermilion'], 'o'),
    'BBB': (OKABE['blue'],      's'),
    'BBC': (OKABE['green'],     'D'),
    'CBA': (OKABE['purple'],    '^'),
    'CBB': (OKABE['orange'],    'v'),
    'CBC': (OKABE['skyblue'],   'P'),
}
BATCH_ORDER = ['BBA', 'BBB', 'BBC', 'CBA', 'CBB', 'CBC']

# --------------------------------------------------------------- colour maps --
DIVERGING = LinearSegmentedColormap.from_list(
    'okabe_div', ['#7A2E00', '#D55E00', '#F0A868', '#FFFFFF',
                  '#8FC1E3', '#0072B2', '#00436B'])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'okabe_seq', ['#FFFFFF', '#CFE3F2', '#8FC1E3', '#3D8FC6',
                  '#0072B2', '#00436B'])
SEQUENTIAL_R = SEQUENTIAL.reversed()

# ------------------------------------------------------------------- fonts ----
# Sized for a two-column journal page: single-column figures are reproduced at
# roughly 45 % and full-width figures at roughly 95 % of these dimensions, so
# the base size has to be generous.
# elsarticle two-column geometry.  Figures are drawn at their FINAL printed
# width so that a point size here is the same point size on the page: a figure
# authored 16 in wide and placed in a 3.4 in column would render 13 pt text at
# under 3 pt.  Author single-column panels at W_COL and spanning panels at
# W_FULL, then use \includegraphics[width=\columnwidth] / [width=\textwidth].
W_COL, W_FULL = 3.45, 7.16          # inches

FS_BASE, FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT = 9.5, 9, 10, 11, 9, 8.5


def apply(base=FS_BASE):
    """Install the shared rcParams.  Call once, at the top of a figure script."""
    plt.rcParams.update({
        'figure.dpi':            110,
        'savefig.dpi':           400,
        'savefig.bbox':          'tight',
        'savefig.facecolor':     'white',
        'figure.facecolor':      'white',
        'font.size':             base,
        'axes.titlesize':        FS_TITLE,
        'axes.titleweight':      'bold',
        'figure.titleweight':    'bold',   # suptitle bold too
        'figure.titlesize':      FS_TITLE,
        'axes.labelsize':        FS_LABEL,
        'xtick.labelsize':       FS_TICK,
        'ytick.labelsize':       FS_TICK,
        'legend.fontsize':       FS_LEGEND,
        'legend.title_fontsize': FS_LEGEND,
        'axes.prop_cycle':       plt.cycler(color=CYCLE),
        'axes.linewidth':        0.9,
        'axes.edgecolor':        '#333333',
        'axes.grid':             True,
        'grid.color':            GRIDCOLOR,
        'grid.linewidth':        0.6,
        'grid.alpha':            0.9,
        'axes.axisbelow':        True,
        'lines.linewidth':       1.5,
        'lines.markersize':      5,
        'lines.markeredgewidth': 0.7,
        'patch.linewidth':       0.8,
        'xtick.major.width':     0.9,
        'ytick.major.width':     0.9,
        'xtick.major.size':      3.2,
        'ytick.major.size':      3.2,
        'legend.frameon':        True,
        'legend.framealpha':     0.95,
        'legend.edgecolor':      '#BBBBBB',
        'mathtext.default':      'regular',
        # constrained layout resolves label/title/colorbar collisions that
        # manual subplots_adjust cannot once fonts are at print size
        'figure.constrained_layout.use':   True,
        'figure.constrained_layout.h_pad': 0.045,
        'figure.constrained_layout.w_pad': 0.045,
        'figure.constrained_layout.hspace': 0.055,
        'figure.constrained_layout.wspace': 0.055,
    })


def batch_style(name):
    """(colour, marker) for a batch label; falls back to grey circles."""
    return BATCH.get(str(name).strip().upper(), (REFERENCE, 'o'))


def title(ax, text, size=None, loc='left'):
    """Every axes title in the paper goes through here: bold, one size."""
    ax.set_title(text, loc=loc, fontweight='bold', fontsize=size or FS_TITLE)


def panel_title(ax, letter, text, **kw):
    """Panel label folded into the title: survives constrained layout, never
    collides with a neighbouring axes the way a floating text label does."""
    ax.set_title(f'({letter}) {text}', loc='left', **kw)


def panel_tag(ax, letter, dx=-0.13, dy=1.06, size=11):
    """Deprecated floating tag; kept so older scripts keep running."""
    ax.text(dx, dy, f'({letter})', transform=ax.transAxes, fontsize=size,
            fontweight='bold', va='top', ha='left')


def save(fig, filename, plots_dir, extra_dirs=()):
    """Write to analysis_plots and mirror into the paper figure directories."""
    out = os.path.join(str(plots_dir), filename)
    fig.savefig(out)
    print('Wrote', out)
    for d in extra_dirs:
        if d and os.path.isdir(str(d)):
            shutil.copy(out, str(d))
    plt.close(fig)
    return out
