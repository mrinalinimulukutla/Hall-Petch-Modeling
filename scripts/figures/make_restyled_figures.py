#!/usr/bin/env python3
"""Regenerate the paper/SI figures that previously had no generator in the repo,
using the shared palette and typography of `_figstyle.py`.

Covers:
  fig02_correlation_matrix   fig_sss_parity          fig_scaling_fits
  fig_scaling_deltabic       fig_bayesian_bma        fig_comp_hp_models_ab
  fig_misfit_scatter         fig_premodel_hulls_heatmap
  fig08_pysr_pareto

Deliberately NOT regenerated:
  fig00_framework_overview   keeps its own scheme by request
  composition_microstructure raw EBSD/BSE micrographs
  tensile_tests              measured stress-strain curves
  fig07_shap_summary         needs xgboost + shap (not installed here)
  fig_bayesian_n             needs the PyMC posterior draws (not archived)

Everything is recomputed from data/derived/data_with_vlc.csv and results/*.csv;
no number is taken from the previous PNGs.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull, Delaunay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from _config import DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_FIG_DIR, REPO_ROOT
import _figstyle as S

S.apply()
MIRROR = (PAPER_FIG_DIR,)

DF = pd.read_csv(f'{DATA_DIR}/data_with_vlc.csv')
YS = DF.dropna(subset=['YS']).reset_index(drop=True)
HV = DF.dropna(subset=['HV']).reset_index(drop=True)
ELEMS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'V']          # Ni is the reference


# --------------------------------------------------------------- utilities --
def loo_bic(X, y):
    """OLS LOO R^2 (hat-matrix shortcut) and BIC for a design matrix with intercept."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    h = np.clip(np.diag(H), 0, 1 - 1e-10)
    e_loo = e / (1 - h)
    ss = ((y - y.mean()) ** 2).sum()
    n, k = len(y), X.shape[1]
    rss = (e ** 2).sum()
    return 1 - (e_loo ** 2).sum() / ss, n * np.log(rss / n) + k * np.log(n), k


def batch_legend(ax, ncol=3, loc='best'):
    handles = [plt.Line2D([], [], color=c, marker=m, ls='', ms=4.5,
                          mec='white', mew=0.5, label=b)
               for b, (c, m) in S.BATCH.items()]
    ax.legend(handles=handles, ncol=ncol, loc=loc, fontsize=S.FS_ANNOT - 0.7,
              handletextpad=0.3, columnspacing=0.7, borderpad=0.35)


def scatter_by_batch(ax, x, y, frame, s=24, alpha=0.9):
    for b in S.BATCH_ORDER:
        m = frame.Iteration.astype(str).str.upper() == b
        if not m.any():
            continue
        c, mk = S.batch_style(b)
        ax.scatter(np.asarray(x)[m.values], np.asarray(y)[m.values], s=s, c=c,
                   marker=mk, edgecolor='white', linewidth=0.5, alpha=alpha,
                   zorder=3, label=b)


# ------------------------------------------------ 1. correlation matrix ------
def fig_correlation_matrix():
    cols = ['YS', 'HV', 'd_inv_sqrt', 'GrainSize', 'SD_GS', 'VEC', 'dH_mix',
            'dS_mix', 'Omega', 'delta_chi', 'delta', 'Phi_VLC', 'eps_Labusch',
            'ColdWork', 'RecrystT', 'HoldTime']
    nice = {'d_inv_sqrt': r'$d^{-1/2}$', 'GrainSize': r'$d$', 'SD_GS': r'SD$_{\rm grain}$',
            'dH_mix': r'$\Delta H_{\rm mix}$', 'dS_mix': r'$\Delta S_{\rm mix}$',
            'Omega': r'$\Omega$', 'delta_chi': r'$\Delta\chi$', 'delta': r'$\delta$',
            'Phi_VLC': r'$\Phi_{\rm VLC}$', 'eps_Labusch': r'$\varepsilon_L$',
            'ColdWork': 'cold work', 'RecrystT': r'$T_{\rm rx}$', 'HoldTime': r'$t_{\rm hold}$'}
    C = DF[cols].corr()
    labels = [nice.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(S.W_FULL, 6.9))
    im = ax.imshow(C.values, cmap=S.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=48, ha='right', fontsize=S.FS_ANNOT + 0.5)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels, fontsize=S.FS_ANNOT + 0.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = C.values[i, j]
            # two significant digits without the leading zero keeps the glyph
            # count low enough to stay legible in a 16 x 16 grid
            lab = f'{v:.2f}'.replace('0.', '.').replace('-.', '\u2212.')
            if lab in ('1.00', '.00'):
                lab = '1' if v > 0.5 else '0'
            ax.text(j, i, lab, ha='center', va='center', fontsize=7.0,
                    fontweight='bold' if abs(v) >= 0.6 else 'normal',
                    color='white' if abs(v) > 0.62 else '#1A1A1A')
    # separate the two targets from the predictors
    ax.axhline(1.5, color='#111111', lw=1.8); ax.axvline(1.5, color='#111111', lw=1.8)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.7)
    ax.tick_params(which='minor', length=0)
    S.title(ax, 'Pearson correlation: targets, microstructure, descriptors',
            size=S.FS_ANNOT + 1.5, loc='center')
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('Pearson $r$', fontsize=S.FS_ANNOT)
    S.save(fig, 'fig02_correlation_matrix.png', PLOTS_DIR, MIRROR)


# ------------------------------------------------------- 2. SSS parity -------
def fig_sss_parity():
    models = [('sigma_y_VLC_300K', 'VLC (uncalibrated)'),
              ('sigma_Labusch', 'Labusch (cal.)'),
              ('sigma_TC', 'Toda-Caraballo (cal.)')]
    sub = YS.dropna(subset=['sigma_0_exp'])
    fig, axes = plt.subplots(1, 3, figsize=(S.W_FULL, 2.75), sharey=True)
    lim = [0, max(sub[[m for m, _ in models]].max().max(),
                  sub.sigma_0_exp.max()) * 1.08]
    for ax, (col, name), tag in zip(axes, models, 'abc'):
        ax.plot(lim, lim, color=S.REFERENCE, ls='--', lw=2, zorder=1)
        scatter_by_batch(ax, sub.sigma_0_exp.values, sub[col].values, sub)
        ratio = sub[col].mean() / sub.sigma_0_exp.mean()
        ax.set_xlim(lim); ax.set_ylim(lim)
        S.title(ax, f'({tag}) {name}', size=S.FS_ANNOT + 1.5)
        ax.text(0.04, 0.94, f'ratio {ratio:.2f}', transform=ax.transAxes,
                va='top', fontsize=S.FS_ANNOT - 0.5, color='#333333',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC', lw=0.6))
        ax.text(0.95, 0.06, '1:1', transform=ax.transAxes, ha='right',
                color=S.REFERENCE, style='italic', fontsize=S.FS_ANNOT - 0.5)
    axes[0].set_ylabel(r'predicted $\sigma_{\rm SSS}$ (MPa)')
    axes[1].set_xlabel(r'Hall–Petch-corrected experimental $\sigma_0$  (MPa)')
    batch_legend(axes[1], ncol=3, loc='lower right')
    S.save(fig, 'fig_sss_parity.png', PLOTS_DIR, MIRROR)


# ----------------------------------------- 3+4. grain-size scaling laws ------
LAWS = [
    (r'$d^{-1/2}$',      lambda d: [d ** -0.5]),
    (r'$d^{-1}$',        lambda d: [d ** -1.0]),
    (r'$d^{-1/3}$',      lambda d: [d ** (-1 / 3)]),
    (r'$d^{-2/3}$',      lambda d: [d ** (-2 / 3)]),
    (r'$\ln(d)/d$',      lambda d: [np.log(d) / d]),
    (r'$\ln d$',         lambda d: [np.log(d)]),
    (r'$d^{-1/2}+d^{-1}$', lambda d: [d ** -0.5, d ** -1.0]),
    (r'$d^{-1}+d^{-2}$', lambda d: [d ** -1.0, d ** -2.0]),
]


def _fit_laws(frame, target):
    d, y = frame.GrainSize.values, frame[target].values
    rows = []
    for name, fn in LAWS:
        r2, bic, k = loo_bic(np.column_stack(fn(d)), y)
        rows.append((name, r2, bic, k))
    # free exponent
    best = max(((n, *loo_bic(np.column_stack([d ** -n]), y))
                for n in np.arange(0.05, 3.001, 0.005)), key=lambda t: t[1])
    rows.append((rf'$d^{{-n}}$, $n$={best[0]:.3f}', best[1], best[2] + np.log(len(y)), best[3] + 1))
    out = pd.DataFrame(rows, columns=['law', 'LOO_R2', 'BIC', 'k'])
    out['dBIC'] = out.BIC - out.BIC.min()
    return out.sort_values('dBIC').reset_index(drop=True)


def fig_scaling():
    tab_ys, tab_hv = _fit_laws(YS, 'YS'), _fit_laws(HV, 'HV')

    # ---- (a) ΔBIC bars -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 3.35))
    for ax, tab, name, unit, tag in ((axes[0], tab_ys, 'Yield strength', 'MPa', 'a'),
                                     (axes[1], tab_hv, 'Vickers hardness', 'HV', 'b')):
        col = [S.LINEAR if v < 2 else S.NONLINEAR for v in tab.dBIC]
        ax.barh(range(len(tab)), tab.dBIC, color=col, edgecolor='#333333', height=0.72)
        ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab.law)
        ax.invert_yaxis()
        ax.axvline(2, color=S.HIGHLIGHT, ls='--', lw=1.8, zorder=2,
                   label=r'$\Delta$BIC = 2')
        ax.legend(loc='upper right', fontsize=S.FS_ANNOT - 0.5,
                  handlelength=1.6, borderpad=0.3, framealpha=0.95)
        for i, (v, r2) in enumerate(zip(tab.dBIC, tab.LOO_R2)):
            ax.text(v + max(tab.dBIC) * 0.02, i, f'{r2:.3f}', va='center',
                    fontsize=S.FS_ANNOT, color='#333333', zorder=6,
                    bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none'))
        ax.set_xlim(0, max(tab.dBIC) * 1.26)
        ax.set_xlabel(r'$\Delta$BIC vs. best law')
        S.title(ax, f'({tag}) {name}', size=S.FS_ANNOT + 1.5)
    fig.suptitle('Grain-size scaling laws — blue: inside the $\\Delta$BIC < 2 band.  '
                 'Numbers beside each bar are LOO $R^2$.', fontsize=S.FS_ANNOT + 1)
    S.save(fig, 'fig_scaling_deltabic.png', PLOTS_DIR, MIRROR)

    # ---- (b) fitted curves -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 3.15))
    for ax, frame, tgt, ylab, tab, tag in (
            (axes[0], YS, 'YS', 'Yield strength (MPa)', tab_ys, 'a'),
            (axes[1], HV, 'HV', 'Vickers hardness', tab_hv, 'b')):
        scatter_by_batch(ax, frame.GrainSize.values, frame[tgt].values, frame, s=20)
        grid = np.linspace(frame.GrainSize.min() * 0.92, frame.GrainSize.max() * 1.05, 300)
        keep = [n for n in tab.law[:3]]
        styles = ['-', '--', ':']
        for name, ls in zip(keep, styles):
            fn = dict((l, f) for l, f in LAWS).get(name)
            if fn is None:
                continue
            Xf = np.column_stack([np.ones(len(frame))] + fn(frame.GrainSize.values))
            beta, *_ = np.linalg.lstsq(Xf, frame[tgt].values, rcond=None)
            Xg = np.column_stack([np.ones(len(grid))] + fn(grid))
            ax.plot(grid, Xg @ beta, ls, color='#111111', lw=2.6, zorder=5, label=name)
        ax.set_xlabel(r'mean grain size $d$  ($\mu$m)')
        ax.set_ylabel(ylab)
        ax.margins(x=0.02)
        ax.legend(loc='upper right', fontsize=S.FS_ANNOT - 0.7,
                  handlelength=1.6, borderpad=0.3, labelspacing=0.25)
        S.title(ax, f'({tag}) {ylab.split(" (")[0]}', size=S.FS_ANNOT + 1.5)
    batch_legend(axes[1], ncol=2, loc='lower left')
    S.save(fig, 'fig_scaling_fits.png', PLOTS_DIR, MIRROR)


# --------------------------------------------- 5. Bayesian model averaging ---
def fig_bayesian_bma():
    t = pd.read_csv(f'{RESULTS_DIR}/bayesian_model_comparison.csv', index_col=0)
    t = t.sort_values('rank')
    fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 2.95))
    ax = axes[0]
    ax.barh(range(len(t)), t.elpd_diff, xerr=t.dse, color=S.LINEAR,
            edgecolor='#333333', height=0.7, error_kw=dict(ecolor='#555555', lw=1.6))
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index); ax.invert_yaxis()
    ax.set_xlabel(r'$\Delta$elpd$_{\rm LOO}$ relative to the best model')
    S.title(ax, '(a) PSIS-LOO ranking', size=S.FS_ANNOT + 1.5)
    ax = axes[1]
    w = t.weight.clip(lower=0)
    ax.barh(range(len(t)), w, color=[S.HIGHLIGHT if x > 0.05 else '#B8D4E8' for x in w],
            edgecolor='#333333', height=0.7)
    ax.set_yticks(range(len(t))); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    for i, x in enumerate(w):
        if x > 0.005:
            ax.text(min(x, 1.0) - 0.03, i, f'{x:.2f}', va='center', ha='right',
                    fontsize=S.FS_ANNOT - 1, color='#5A3E00', fontweight='bold')
    ax.text(0.97, 0.04, 'all remaining weights < 0.005', transform=ax.transAxes,
            ha='right', fontsize=S.FS_ANNOT - 1.5, color='#666666', style='italic')
    ax.set_xlabel('stacking weight')
    S.title(ax, '(b) Model-averaging weight', size=S.FS_ANNOT + 1.5)
    fig.suptitle('Bayesian comparison of grain-size scaling laws (yield strength)',
                 fontsize=S.FS_ANNOT + 1)
    S.save(fig, 'fig_bayesian_bma.png', PLOTS_DIR, MIRROR)


# --------------------------------------- 6. composition Hall–Petch models ----
def fig_comp_hp_models():
    """Family 3 hierarchy: LOO *and* LOBO side by side, plus dBIC vs M3.

    M15 contains both additive SD_grain and SD_grain*d^-1/2 terms. The
    additive-only control is documented in the manuscript and supplement,
    rather than plotted as another hierarchy row.
    """
    from sklearn.linear_model import LinearRegression
    d = YS.d_inv_sqrt.values
    SD = YS.SD_GS.values
    F = {e: YS[f'{e}_frac'].values for e in ELEMS}
    y = YS.YS.values
    grp = YS.Iteration.values

    specs = {
        'M0: baseline HP':            np.column_stack([d]),
        'M1: $\\sigma_0$(V)':          np.column_stack([F['V'], d]),
        'M3: $\\sigma_0$(all 7)':      np.column_stack([*[F[e] for e in ELEMS], d]),
        'M4: $k$(V)':                 np.column_stack([d, F['V'] * d]),
        'M6: $k$(all 7)':             np.column_stack([d, *[F[e] * d for e in ELEMS]]),
        'M10: $\\sigma_0$+$k$(all)':   np.column_stack([*[F[e] for e in ELEMS], d,
                                                       *[F[e] * d for e in ELEMS]]),
        'M11: $\\sigma_0(\\delta)$':    np.column_stack([YS.delta.values, d]),
        'M15: M3 $+$ SD terms':
                                      np.column_stack([*[F[e] for e in ELEMS], d, SD, SD * d]),
    }

    def q2(a, b):
        return 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()

    def lobo(X):
        p = np.zeros(len(y))
        for k in np.unique(grp):
            te = grp == k
            p[te] = LinearRegression().fit(X[~te], y[~te]).predict(X[te])
        return q2(y, p)

    rows = []
    for n, X in specs.items():
        r2, b, _ = loo_bic(X, y)
        rows.append((n, r2, lobo(X), b))
    t = pd.DataFrame(rows, columns=['model', 'LOO', 'LOBO', 'BIC'])
    t['dBIC'] = t.BIC - t.loc[t.model.str.startswith('M3'), 'BIC'].iloc[0]
    t = t.sort_values('LOO', ascending=False).reset_index(drop=True)
    SDROWS = t.model.str.contains('SD')

    fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 3.5))

    # ---- (a) LOO and LOBO as paired bars ----------------------------------
    ax = axes[0]
    yy = np.arange(len(t))
    h = 0.38
    ax.barh(yy - h / 2, t.LOO, height=h, color=S.LINEAR,
            edgecolor='#333333', label='LOO')
    ax.barh(yy + h / 2, t.LOBO, height=h, color=S.OKABE['skyblue'],
            edgecolor='#333333', label='LOBO')
    for i, (a, b_) in enumerate(zip(t.LOO, t.LOBO)):
        ax.text(a - 0.012, i - h / 2, f'{a:.3f}', va='center', ha='right',
                fontsize=S.FS_ANNOT - 1.5, color='white', fontweight='bold')
        ax.text(b_ - 0.012, i + h / 2, f'{b_:.3f}', va='center', ha='right',
                fontsize=S.FS_ANNOT - 1.5, color='#12405C', fontweight='bold')
    ax.set_yticks(yy)
    ax.set_yticklabels(t.model)
    for i, flag in enumerate(SDROWS):
        if flag:
            ax.get_yticklabels()[i].set_color(S.NONLINEAR)
            ax.get_yticklabels()[i].set_fontweight('bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 0.80)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel('cross-validated $R^2$')
    ax.legend(loc='lower right', fontsize=S.FS_ANNOT - 1, handlelength=1.3,
              borderpad=0.3, labelspacing=0.25)
    S.title(ax, '(a) Predictive ranking', size=S.FS_ANNOT + 1.5)

    # ---- (b) dBIC relative to M3, signed ---------------------------------
    ax = axes[1]
    ts = t.sort_values('dBIC').reset_index(drop=True)
    col = [S.HIGHLIGHT if v < -2 else (S.LINEAR if v < 2 else S.NONLINEAR)
           for v in ts.dBIC]
    ax.barh(np.arange(len(ts)), ts.dBIC, color=col, edgecolor='#333333', height=0.7)
    ax.set_yticks(np.arange(len(ts)))
    ax.set_yticklabels(ts.model)
    for i, flag in enumerate(ts.model.str.contains('SD')):
        if flag:
            ax.get_yticklabels()[i].set_color(S.NONLINEAR)
            ax.get_yticklabels()[i].set_fontweight('bold')
    ax.invert_yaxis()
    ax.axvline(0, color='#111111', lw=1.2)
    for lim in (-2, 2):                      # the +-2 'indistinguishable' band
        ax.axvline(lim, color=S.REFERENCE, ls=':', lw=0.9, zorder=2)
    for i, v in enumerate(ts.dBIC):
        off = -1.8 if v < 0 else 1.8
        ax.text(v + off, i, f'{v:+.1f}', va='center',
                ha='right' if v < 0 else 'left',
                fontsize=S.FS_ANNOT - 1.5, color='#333333', zorder=6,
                bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none'))
    ax.set_xlim(-38, 48)
    ax.set_xlabel(r'$\Delta$BIC vs. M3')
    S.title(ax, '(b) Parsimony', size=S.FS_ANNOT + 1.5)

    fig.suptitle('Composition-dependent Hall–Petch hierarchy for yield strength.  '
                 'Negative $\\Delta$BIC favours the model over M3.',
                 fontsize=S.FS_ANNOT + 0.5)
    S.save(fig, 'fig_comp_hp_models_ab.png', PLOTS_DIR, MIRROR)


# ------------------------------------------------- 7. misfit vs alpha_i ------
def fig_misfit():
    # FCC lattice parameters as documented in scripts/00_data_preparation/
    # vlc_corrected.py (Fe = 3.590 A is gamma-Fe; see CLAUDE.md 7a).  Defined
    # locally on purpose: importing the analysis module would re-run it and
    # overwrite data/derived/data_with_vlc.csv.
    A_FCC = {'Al': 4.050, 'Co': 3.545, 'Cr': 3.520, 'Cu': 3.615,
             'Fe': 3.590, 'Mn': 3.540, 'Ni': 3.524, 'V': 3.720}
    ATOMIC_VOL = {el: (a * 1e-10) ** 3 / 4 for el, a in A_FCC.items()}
    coef = pd.read_csv(f'{RESULTS_DIR}/m3_coefficients.csv')
    alpha = {r.coefficient.replace('alpha_', ''): r.value
             for _, r in coef.iterrows() if r.coefficient.startswith('alpha_')}
    err = {r.coefficient.replace('alpha_', ''): (r.hi95 - r.lo95) / 2
           for _, r in coef.iterrows() if r.coefficient.startswith('alpha_')}
    mean_frac = {e: DF[f'{e}_frac'].mean() for e in ATOMIC_VOL}
    V_bar = sum(mean_frac[e] * ATOMIC_VOL[e] for e in ATOMIC_VOL)
    misfit = {e: abs(ATOMIC_VOL[e] - V_bar) / V_bar for e in ATOMIC_VOL}

    fig, ax = plt.subplots(figsize=(S.W_COL, 3.05))
    pts = sorted(((misfit[e] * 100, alpha[e], e) for e in ELEMS if e in misfit),
                 key=lambda t: t[1])
    # alternate the label side and stagger vertically: 7 points in a tight
    # cluster otherwise print on top of one another
    offsets = {'V': (9, 4), 'Mn': (-6, 9), 'Al': (-24, 4), 'Co': (9, -1),
               'Cr': (9, -9), 'Cu': (-22, -6), 'Fe': (9, 5)}
    for x, y, e in pts:
        big = DF[f'{e}_frac'].max() - DF[f'{e}_frac'].min() < 0.05
        c = S.HIGHLIGHT if e == 'V' else ('#999999' if big else S.LINEAR)
        ax.errorbar(x, y, yerr=err[e], fmt='o', ms=4.5, color=c, ecolor='#999999',
                    elinewidth=0.9, capsize=2.2, mec='white', mew=0.7, zorder=3)
        ax.annotate(e, (x, y), textcoords='offset points',
                    xytext=offsets.get(e, (9, 4)),
                    fontsize=S.FS_ANNOT, fontweight='bold', color=c, zorder=4)
    ax.axhline(0, color=S.REFERENCE, lw=1.8, ls='--')
    ax.set_xlabel(r'fractional volume misfit $|\Delta V_i/\bar{V}|$  (%)')
    ax.set_ylabel(r'$\alpha_i$  (MPa, Ni reference)')
    S.title(ax, 'M3 coefficients vs. atomic-volume misfit', size=S.FS_ANNOT + 1.5, loc='center')
    ax.text(0.97, 0.05, 'orange: V\ngrey: narrow range',
            transform=ax.transAxes, ha='right', fontsize=S.FS_ANNOT - 1.2,
            color='#444444', bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                       ec='#CCCCCC', lw=0.6))
    ax.margins(x=0.16, y=0.16)
    S.save(fig, 'fig_misfit_scatter.png', PLOTS_DIR, MIRROR)


# ------------------------------------------ 8. batch composition-hull map ----
def fig_hulls():
    X = StandardScaler().fit_transform(DF[[f'{e}_frac' for e in
                                           ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']]])
    P = PCA(n_components=2).fit(X)
    Z = P.transform(X)
    groups = {b: Z[(DF.Iteration.astype(str).str.upper() == b).values] for b in S.BATCH_ORDER}
    M = np.full((6, 6), np.nan)
    for i, a in enumerate(S.BATCH_ORDER):
        for j, b in enumerate(S.BATCH_ORDER):
            if a == b or len(groups[b]) < 3 or len(groups[a]) == 0:
                continue
            try:
                tri = Delaunay(groups[b])
                M[i, j] = (tri.find_simplex(groups[a]) >= 0).mean()
            except Exception:
                pass

    fig, axes = plt.subplots(1, 2, figsize=(S.W_FULL, 3.3))
    ax = axes[0]
    for b in S.BATCH_ORDER:
        g = groups[b]
        c, mk = S.batch_style(b)
        ax.scatter(g[:, 0], g[:, 1], s=22, c=c, marker=mk, edgecolor='white',
                   linewidth=0.5, zorder=3, label=b)
        if len(g) >= 3:
            h = ConvexHull(g)
            pts = np.vstack([g[h.vertices], g[h.vertices][:1]])
            ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.2, alpha=0.85, zorder=2)
            ax.fill(pts[:, 0], pts[:, 1], color=c, alpha=0.10, zorder=1)
    ax.set_xlabel(f'PC1  ({P.explained_variance_ratio_[0]*100:.0f}% of composition variance)')
    ax.set_ylabel(f'PC2  ({P.explained_variance_ratio_[1]*100:.0f}%)')
    S.title(ax, '(a) Composition coverage by batch', size=S.FS_ANNOT + 1.5)
    ax.legend(ncol=2, loc='best', fontsize=S.FS_ANNOT - 1, handletextpad=0.3,
              columnspacing=0.7, borderpad=0.3)

    ax = axes[1]
    im = ax.imshow(M, cmap=S.SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(S.BATCH_ORDER)
    ax.set_yticks(range(6)); ax.set_yticklabels(S.BATCH_ORDER)
    ax.set_xlabel('training hull (batch)'); ax.set_ylabel('held-out batch')
    ax.grid(False)
    for i in range(6):
        for j in range(6):
            if i == j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor='#EDEDED',
                                       edgecolor='white', lw=1.5))
                continue
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=S.FS_ANNOT,
                        color='white' if v > 0.60 else '#222222')
    mean_off = np.nanmean(M[~np.eye(6, dtype=bool)])
    S.title(ax, f'(b) Pairwise hull containment (mean {mean_off:.2f})', size=S.FS_ANNOT + 1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('fraction inside hull', fontsize=S.FS_ANNOT - 1)
    S.save(fig, 'fig_premodel_hulls_heatmap.png', PLOTS_DIR, MIRROR)


# ------------------------------------------------- 9. PySR Pareto front ------
def fig_pysr_pareto():
    p = pd.read_csv(f'{RESULTS_DIR}/pysr_pareto_full.csv').sort_values('complexity')
    fig, ax = plt.subplots(figsize=(S.W_COL, 2.95))
    ax.plot(p.complexity, p.loss, '-o', color=S.LINEAR, ms=4, lw=1.4,
            mec='white', mew=1.2, zorder=3, label='Pareto front')
    ax.set_yscale('log')
    ax.set_xlabel('equation complexity (nodes)')
    ax.set_ylabel('training loss (MSE)')
    S.title(ax, 'PySR complexity–loss front (YS)', size=S.FS_ANNOT + 1.5, loc='center')
    if 'score' in p.columns and p.score.notna().any():
        elbow = p.loc[p.score.idxmax()]
        ax.scatter([elbow.complexity], [elbow.loss], s=120, marker='*',
                   color=S.HIGHLIGHT, edgecolor='#333333', linewidth=1.4, zorder=5,
                   label='elbow (parsimony knee)')
    acc = p.iloc[-1]
    ax.scatter([acc.complexity], [acc.loss], s=55, marker='D',
               color=S.NONLINEAR, edgecolor='#333333', linewidth=1.4, zorder=5,
               label='accuracy (lowest loss)')
    ax.legend(loc='upper right', fontsize=S.FS_ANNOT - 1, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.25)
    ax.text(0.03, 0.05, 'selection made after viewing\nthis front $\\Rightarrow$ post-selection',
            transform=ax.transAxes, fontsize=S.FS_ANNOT - 1.2, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', lw=0.6))
    S.save(fig, 'fig08_pysr_pareto.png', PLOTS_DIR, MIRROR)


if __name__ == '__main__':
    fig_correlation_matrix()
    fig_sss_parity()
    fig_scaling()
    fig_bayesian_bma()
    fig_comp_hp_models()
    fig_misfit()
    fig_hulls()
    fig_pysr_pareto()
    print('\nAll restyled figures written.')
