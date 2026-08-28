#!/usr/bin/env python3
"""Regenerate the batch-colored paper figures with a colorblind-safe scheme.

Writes (to paper/figures/):
  fig01_hall_petch.png     classical Hall-Petch, YS and HV panels
  fig09_tabor_relation.png four-panel Tabor / C_eff analysis
  fig_external_ab.png      external-validation parity, SISSO Full vs Robust

Colors follow the Okabe-Ito palette and every batch/source additionally has
a distinct marker shape, so groups remain separable under deuteranopia,
protanopia, and in grayscale. Reads only cached CSVs; part of `make figures`.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import DATA_DIR, RESULTS_DIR, PAPER_FIG_DIR
import _figstyle as S
S.apply()

# Okabe-Ito palette; hue associations kept close to the legacy palette.
BATCH_STYLE = S.BATCH
SOURCE_STYLE = {
    'Citrine':       ('#0072B2', 'o'),
    'Schneider2021': ('#D55E00', 's'),
    'Otto2013':      ('#009E73', 'D'),
    'Huang2019':     ('#CC79A7', '^'),
}
HV_TO_MPA = 9.80665

df = pd.read_csv(f'{DATA_DIR}/data_with_descriptors.csv')
df['dinv'] = 1.0 / np.sqrt(df['GrainSize'])

def batch_scatter(ax, x, y, sub, batch):
    c, m = BATCH_STYLE[batch]
    ax.scatter(sub[x], sub[y], s=17, facecolor=c, edgecolor='white',
               lw=0.4, marker=m, label=batch, alpha=0.92, zorder=3)

# ---------------- fig01: classical Hall-Petch (YS | HV) ---------------------
fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 3.4))
for ax, target, unit, name, tag in [(axes[0], 'YS', 'MPa', 'Yield strength', 'a'),
                                    (axes[1], 'HV', 'HV', 'Hardness', 'b')]:
    d = df.dropna(subset=[target, 'GrainSize'])
    for b in BATCH_STYLE:
        sub = d[d['Iteration'] == b]
        if len(sub): batch_scatter(ax, 'dinv', target, sub, b)
    k, b0 = np.polyfit(d['dinv'], d[target], 1)
    r2 = 1 - np.sum((d[target] - (b0 + k * d['dinv']))**2) / np.sum((d[target] - d[target].mean())**2)
    xs = np.linspace(d['dinv'].min(), d['dinv'].max(), 50)
    sym = 'σ₀' if target == 'YS' else 'H₀'
    ax.plot(xs, b0 + k * xs, 'k--', lw=1.3,
            label=f'fit, R²={r2:.3f}')
    ax.text(0.03, 0.96, f'{sym}={b0:.0f} + {k:.0f}·d⁻¹ᐟ²', transform=ax.transAxes,
            va='top', fontsize=S.FS_ANNOT - 1, color='#333333')
    ax.set_xlabel('d⁻¹ᐟ² (μm⁻¹ᐟ²)')
    ax.set_ylabel('YS (MPa)' if target == 'YS' else 'HV')
    ax.set_title(f'({tag}) Hall–Petch: {name}', loc='left', fontsize=S.FS_ANNOT + 1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=4,
              fontsize=S.FS_ANNOT - 1.5, handletextpad=0.3, columnspacing=0.8,
              borderpad=0.3, frameon=False)
    ax.margins(0.05)
plt.savefig(os.path.join(str(PAPER_FIG_DIR), 'fig01_hall_petch.png'), dpi=200,
            bbox_inches='tight')
plt.close()
print('Wrote fig01_hall_petch.png')

# ---------------- fig09: Tabor relation (4 panels) ---------------------------
d = df.dropna(subset=['YS', 'HV', 'GrainSize']).copy()
d['HV_MPa'] = d['HV'] * HV_TO_MPA
d['Ceff'] = d['HV_MPa'] / d['YS']
fig, axes = plt.subplots(2, 2, figsize=(S.W_FULL, 5.6))

ax = axes[0, 0]
for b in BATCH_STYLE:
    sub = d[d['Iteration'] == b]
    if len(sub): batch_scatter(ax, 'YS', 'HV_MPa', sub, b)
C_fit = np.sum(d['YS'] * d['HV_MPa']) / np.sum(d['YS']**2)
xs = np.linspace(d['YS'].min(), d['YS'].max() * 1.1, 50)
ax.plot(xs, 3 * xs, 'k--', lw=2, label='C = 3 (Tabor)')
ax.plot(xs, C_fit * xs, color='#E69F00', lw=2.5, label=f'Best fit (C = {C_fit:.2f})')
ax.set_xlabel('Yield Strength (MPa)'); ax.set_ylabel('HV (MPa)')
ax.set_title('(a) Tabor Relation: HV vs YS'); ax.legend(); ax.grid(alpha=0.9)

ax = axes[0, 1]
ax.hist(d['Ceff'], bins=22, density=True, facecolor='#56B4E9',
        edgecolor='#333', lw=0.8)
ax.axvline(3, color='#D55E00', ls='--', lw=2.2, label='C = 3 (Tabor)')
ax.axvline(d['Ceff'].mean(), color='#000000', lw=2.2,
           label=f'Mean = {d["Ceff"].mean():.2f}')
ax.set_xlabel('C_eff = HV_MPa / YS'); ax.set_ylabel('Density')
ax.set_title('(b) Distribution of Effective Tabor Factor')
ax.legend(); ax.grid(alpha=0.9)

ax = axes[1, 0]
sc = ax.scatter(d['dinv'], d['Ceff'], c=d['V'], cmap='cividis', s=46,
                edgecolor='#333', lw=0.6)
k, b0 = np.polyfit(d['dinv'], d['Ceff'], 1)
r = np.corrcoef(d['dinv'], d['Ceff'])[0, 1]
xs = np.linspace(d['dinv'].min(), d['dinv'].max(), 40)
ax.plot(xs, b0 + k * xs, color='#000000', lw=2)
plt.colorbar(sc, ax=ax, label='V fraction')
ax.set_xlabel('d⁻¹ᐟ² (μm⁻¹ᐟ²)'); ax.set_ylabel('C_eff')
ax.set_title(f'(c) C_eff vs Grain Size (r = {r:.3f})'); ax.grid(alpha=0.9)

ax = axes[1, 1]
from scipy import stats as st
r, pval = st.pearsonr(d['V'], d['Ceff'])
ax.scatter(d['V'], d['Ceff'], s=46, facecolor='#0072B2', edgecolor='#333',
           lw=0.6, alpha=0.85)
k, b0 = np.polyfit(d['V'], d['Ceff'], 1)
xs = np.linspace(0, d['V'].max() * 1.02, 40)
ax.plot(xs, b0 + k * xs, color='#000000', lw=2)
ax.set_xlabel('V fraction'); ax.set_ylabel('C_eff')
ax.set_title(f'(d) C_eff vs V (r = {r:.3f}, p = {pval:.4f})'); ax.grid(alpha=0.9)
plt.savefig(os.path.join(str(PAPER_FIG_DIR), 'fig09_tabor_relation.png'), dpi=200,
            bbox_inches='tight')
plt.close()
print('Wrote fig09_tabor_relation.png')

# ---------------- fig_external_ab: parity, SISSO Full | Robust ---------------
ext = pd.read_csv(f'{RESULTS_DIR}/external_validation_results.csv')
fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 3.3))
for ax, col, name in [(axes[0], 'YS_SISSO', 'SISSO Full'),
                      (axes[1], 'YS_SISSO_robust', 'SISSO Robust')]:
    e = ext.dropna(subset=[col, 'YS_exp'])
    res = e[col] - e['YS_exp']
    rmse = float(np.sqrt(np.mean(res**2)))
    r2 = 1 - np.sum(res**2) / np.sum((e['YS_exp'] - e['YS_exp'].mean())**2)
    for src, (c, m) in SOURCE_STYLE.items():
        sub = e[e['source'] == src]
        if len(sub):
            ax.scatter(sub['YS_exp'], sub[col], s=52, facecolor=c,
                       edgecolor='#333', lw=0.7, marker=m, label=src, alpha=0.9)
    conv = e[e['is_hv_converted'] == True]
    ax.scatter(conv['YS_exp'], conv[col], s=68, facecolor='none', marker='x',
               color='#000000', lw=1.1, label='HV-converted')
    lim = [min(e['YS_exp'].min(), e[col].min()) * 0.9,
           max(e['YS_exp'].max(), e[col].max()) * 1.05]
    xs = np.linspace(lim[0], lim[1], 20)
    ax.plot(xs, xs, color='#666', ls='--', lw=1.6, label='y=x')
    ax.plot(xs, xs + rmse, color='#aaa', ls=':', lw=1.1)
    ax.plot(xs, xs - rmse, color='#aaa', ls=':', lw=1.1)
    ax.text(0.04, 0.94, f'R² = {r2:.3f}\nRMSE = {rmse:.1f} MPa',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='#fdf3dc', edgecolor='#333'))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Experimental YS (MPa)'); ax.set_ylabel('Predicted YS (MPa)')
    ax.set_title(f'{name} — External Validation')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.9)
plt.savefig(os.path.join(str(PAPER_FIG_DIR), 'fig_external_ab.png'), dpi=200,
            bbox_inches='tight')
plt.close()
print('Wrote fig_external_ab.png')
