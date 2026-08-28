#!/usr/bin/env python3
"""Regression tests locking the manuscript's canonical values.

Every assertion here recomputes the quantity from data/derived or reads it
from results/ and compares against the number printed in paper/main.tex or
paper/supplementary.tex. If an analysis script is re-run and a headline value
moves, these tests fail before the manuscript can silently drift.

Run:  pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from _config import DATA_DIR, RESULTS_DIR, PAPER_DIR      # noqa: E402

TOL = 0.002          # tolerance on cross-validated R^2
ELEMS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'V']


# ------------------------------------------------------------- fixtures ----
@pytest.fixture(scope='module')
def df():
    return pd.read_csv(DATA_DIR / 'data_with_vlc.csv')


@pytest.fixture(scope='module')
def ys(df):
    return df.dropna(subset=['YS']).reset_index(drop=True)


@pytest.fixture(scope='module')
def hv(df):
    return df.dropna(subset=['HV']).reset_index(drop=True)


def q2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def scores(X, y, groups):
    """Pooled 5-fold / LOO / LOBO Q^2 and BIC for an OLS design matrix."""
    p5 = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
        p5[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ beta
    h = np.clip(np.diag(X1 @ np.linalg.pinv(X1.T @ X1) @ X1.T), 0, 1 - 1e-10)
    pl = np.zeros(len(y))
    for k in np.unique(groups):
        te = groups == k
        pl[te] = LinearRegression().fit(X[~te], y[~te]).predict(X[te])
    n, k = len(y), X1.shape[1]
    return (q2(y, p5), q2(y, y - e / (1 - h)), q2(y, pl),
            n * np.log((e ** 2).sum() / n) + k * np.log(n))


def design(frame, terms):
    d = frame.d_inv_sqrt.values
    SD = frame.SD_GS.values
    F = {e: frame[f'{e}_frac'].values for e in ELEMS}
    cols = []
    for t in terms:
        if t == 'd':
            cols.append(d)
        elif t == 'SD':
            cols.append(SD)
        elif t == 'SDxd':
            cols.append(SD * d)
        elif t == 'comp':
            cols.extend(F[e] for e in ELEMS)
        elif t == 'V':
            cols.append(F['V'])
        elif t == 'delta':
            cols.append(frame.delta.values)
    return np.column_stack(cols)


# --------------------------------------------------------------- dataset ---
def test_dataset_shape(df):
    assert len(df) == 94
    assert df.YS.notna().sum() == 93
    assert df.HV.notna().sum() == 94


def test_unique_compositions(df):
    key = df[['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']].round(6) \
            .astype(str).agg('|'.join, axis=1)
    assert key.nunique() == 82
    vc = key.value_counts()
    assert (vc > 1).sum() == 9                 # nine repeated chemistries
    assert vc[vc > 1].sum() == 21              # covering 21 records


def test_measurement_ranges(df):
    assert (df.GrainSize.min(), df.GrainSize.max()) == pytest.approx((14.66, 211.75), abs=0.01)
    assert (df.YS.min(), df.YS.max()) == pytest.approx((151.5, 544.5), abs=0.05)
    assert df.GrainSize.corr(df.SD_GS) == pytest.approx(0.801, abs=0.002)


# -------------------------------------------------------------- Family 1 ---
def test_family1_yield_strength(ys):
    s5, loo, lobo, _ = scores(design(ys, ['d']), ys.YS.values, ys.Iteration.values)
    assert s5 == pytest.approx(0.405, abs=TOL)
    assert loo == pytest.approx(0.406, abs=TOL)
    assert lobo == pytest.approx(0.373, abs=TOL)


def test_family1_hardness_fails_cross_batch(hv):
    s5, loo, lobo, _ = scores(design(hv, ['d']), hv.HV.values, hv.Iteration.values)
    assert s5 == pytest.approx(0.086, abs=TOL)
    assert loo == pytest.approx(0.136, abs=TOL)
    assert lobo == pytest.approx(-0.077, abs=TOL)
    assert lobo < 0, 'mean grain size must not transfer for hardness'


# -------------------------------------------------------------- Family 3 ---
@pytest.mark.parametrize('name,terms,expect', [
    ('M0',  ['d'],                        (0.406, 0.373)),
    ('M1',  ['V', 'd'],                   (0.605, 0.584)),
    ('M3',  ['comp', 'd'],                (0.652, 0.625)),
    ('M11', ['delta', 'd'],               (0.456, 0.392)),
    ('M13', ['comp', 'd', 'SD'],          (0.668, 0.595)),
    ('M15', ['comp', 'd', 'SD', 'SDxd'],  (0.694, 0.694)),
])
def test_m_model_hierarchy(ys, name, terms, expect):
    _, loo, lobo, _ = scores(design(ys, terms), ys.YS.values, ys.Iteration.values)
    assert loo == pytest.approx(expect[0], abs=TOL), f'{name} LOO'
    assert lobo == pytest.approx(expect[1], abs=TOL), f'{name} LOBO'


def test_m15_is_the_strongest_verified_low_complexity_model(ys):
    y, g = ys.YS.values, ys.Iteration.values
    _, _, lobo_m3, bic_m3 = scores(design(ys, ['comp', 'd']), y, g)
    s5, loo, lobo, bic = scores(design(ys, ['comp', 'd', 'SD', 'SDxd']), y, g)
    assert s5 == pytest.approx(0.731, abs=TOL)
    assert loo == pytest.approx(0.694, abs=TOL)
    assert lobo == pytest.approx(0.694, abs=TOL)
    assert bic - bic_m3 == pytest.approx(-20.1, abs=0.3)
    assert lobo > lobo_m3, 'M15 must beat M3 under batch-held-out validation'


def test_additive_sd_is_the_control_not_the_result(ys):
    """The additive term gains at LOO and loses at LOBO; the interaction does not."""
    y, g = ys.YS.values, ys.Iteration.values
    _, loo_m3, lobo_m3, _ = scores(design(ys, ['comp', 'd']), y, g)
    _, loo_13, lobo_13, _ = scores(design(ys, ['comp', 'd', 'SD']), y, g)
    _, loo_15, lobo_15, _ = scores(design(ys, ['comp', 'd', 'SD', 'SDxd']), y, g)
    assert loo_13 > loo_m3 and lobo_13 < lobo_m3, 'M13 should trade LOBO for LOO'
    assert lobo_15 > lobo_13, 'the interaction, not the additive term, carries the gain'


def test_m3_coefficients_match_the_manuscript():
    c = pd.read_csv(RESULTS_DIR / 'm3_coefficients.csv').set_index('coefficient')
    assert c.loc['alpha_V', 'value'] == pytest.approx(291.3, abs=1.0)
    assert c.loc['k_HP', 'value'] == pytest.approx(765.8, abs=1.0)


# --------------------------------------------------- external stress test ---
def test_no_model_transfers_to_the_literature_set():
    e = pd.read_csv(RESULTS_DIR / 'external_validation_by_evidence_tier.csv')
    direct = e[e.evidence_tier == 'Tier 1']
    assert len(direct) >= 3
    assert (direct.R2 < 0).all(), 'every direct-measurement R^2 must be negative'
    assert (direct.n == 54).all()


# ---------------------------------------------------------------- Tabor ----
def test_tabor_ratio(df):
    p = df.dropna(subset=['YS', 'HV'])
    c = p.HV.values * 9.807 / p.YS.values
    assert len(p) == 93
    assert c.mean() == pytest.approx(5.13, abs=0.02)
    assert c.std(ddof=1) == pytest.approx(1.36, abs=0.02)


# ------------------------------------------------------ manuscript wiring ---
def test_every_float_is_referenced_in_the_text():
    import re
    for name in ('main.tex', 'supplementary.tex'):
        t = (PAPER_DIR / name).read_text(encoding='utf-8')
        body = re.sub(r'\\caption\{(?:[^{}]|\{[^{}]*\})*\}', '', t)
        labels = {m.group(2) for m in re.finditer(
            r'\\begin\{(figure\*?|table\*?)\}.*?\\label\{([^}]*)\}.*?\\end\{\1\}',
            t, flags=re.S)}
        cited = set(re.findall(r'\\(?:auto)?ref\{([^}]*)\}', body))
        missing = labels - cited
        assert not missing, f'{name}: floats never called in the text: {sorted(missing)}'


def test_manuscript_figures_exist():
    import re
    for name in ('main.tex', 'supplementary.tex'):
        t = (PAPER_DIR / name).read_text(encoding='utf-8')
        for f in set(re.findall(r'includegraphics\[[^]]*\]\{([^}]*)\}', t)):
            assert (PAPER_DIR / 'figures' / f).exists(), f'{name} needs missing figure {f}'
