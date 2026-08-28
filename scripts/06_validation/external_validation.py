#!/usr/bin/env python3
"""
External Validation of SISSO, M3, and XGBoost Models
=====================================================

Tests our Hall-Petch models against independent experimental data:
  1. Citrine/Borg MPEA Dataset (GitHub, ~730 entries)
  2. Schneider benchmarks (CrCoNi, CrFeNi, MnFeNi)
  3. Otto et al. 2013 (CoCrFeMnNi Cantor alloy, 3 grain sizes)
  4. Tsai et al. 2019 (HV-based, converted via C_eff=5.13)

Usage:
    python -u external_validation.py
"""

import re
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# SECTION 0: CONSTANTS & ELEMENTAL PROPERTIES
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f'{PLOTS_DIR}'
DATA_CSV = f'{DATA_DIR}/data_with_vlc.csv'

ELEMENTS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']

# Goldschmidt metallic radii (pm)
RADII = {'Al': 143, 'Co': 125, 'Cr': 128, 'Cu': 128, 'Fe': 126,
         'Mn': 127, 'Ni': 124, 'V': 134}

# Valence electron concentration
VEC_VALS = {'Al': 3, 'Co': 9, 'Cr': 6, 'Cu': 11, 'Fe': 8,
            'Mn': 7, 'Ni': 10, 'V': 5}

# Pauling electronegativity
EN = {'Al': 1.61, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.90, 'Fe': 1.83,
      'Mn': 1.55, 'Ni': 1.91, 'V': 1.63}

# Melting points (K)
TM = {'Al': 933, 'Co': 1768, 'Cr': 2180, 'Cu': 1358, 'Fe': 1811,
      'Mn': 1519, 'Ni': 1728, 'V': 2183}

# Shear modulus (GPa) — Co: 75 GPa (polycrystalline FCC)
SHEAR_MOD = {'Al': 26, 'Co': 75, 'Cr': 115, 'Cu': 48, 'Fe': 82,
             'Mn': 79, 'Ni': 76, 'V': 47}

# Bulk modulus (GPa)
BULK_MOD = {'Al': 76, 'Co': 180, 'Cr': 160, 'Cu': 140, 'Fe': 170,
            'Mn': 120, 'Ni': 180, 'V': 158}

# FCC lattice parameters (Angstrom) — Fe: 3.590 Å (gamma-Fe FCC)
A_FCC = {'Al': 4.050, 'Co': 3.545, 'Cr': 3.520, 'Cu': 3.615,
         'Fe': 3.590, 'Mn': 3.540, 'Ni': 3.524, 'V': 3.720}

# Atomic mass (g/mol)
MASS = {'Al': 26.98, 'Co': 58.93, 'Cr': 52.00, 'Cu': 63.55, 'Fe': 55.85,
        'Mn': 54.94, 'Ni': 58.69, 'V': 50.94}

# Miedema binary mixing enthalpies (kJ/mol) — Takeuchi & Inoue (2005)
HMIX = {
    ('Al', 'Co'): -19, ('Al', 'Cr'): -10, ('Al', 'Cu'): -1, ('Al', 'Fe'): -11,
    ('Al', 'Mn'): -19, ('Al', 'Ni'): -22, ('Al', 'V'): -16,
    ('Co', 'Cr'): -4,  ('Co', 'Cu'): 6,   ('Co', 'Fe'): -1, ('Co', 'Mn'): -5,
    ('Co', 'Ni'): 0,   ('Co', 'V'): -14,
    ('Cr', 'Cu'): 12,  ('Cr', 'Fe'): -1,  ('Cr', 'Mn'): 2,  ('Cr', 'Ni'): -7,
    ('Cr', 'V'): -2,
    ('Cu', 'Fe'): 13,  ('Cu', 'Mn'): 4,   ('Cu', 'Ni'): 4,  ('Cu', 'V'): 5,
    ('Fe', 'Mn'): 0,   ('Fe', 'Ni'): -2,  ('Fe', 'V'): -7,
    ('Mn', 'Ni'): -8,  ('Mn', 'V'): -1,
    ('Ni', 'V'): -18,
}

R_GAS = 8.314  # J/(mol·K)

# Hardness-to-YS conversion factor
C_EFF = 5.13

# Colors for sources
SOURCE_COLORS = {
    'Citrine': '#1f77b4',
    'Schneider': '#ff7f0e',
    'Otto2013': '#2ca02c',
    'Tsai2019': '#d62728',
    'Training': '#999999',
}

print("=" * 70)
print("EXTERNAL VALIDATION OF HALL-PETCH MODELS")
print("=" * 70)


# ============================================================
# SECTION 1: FEATURE COMPUTATION FUNCTIONS
# ============================================================

def compute_oliynyk_features(fracs):
    """Compute Oliynyk-style statistical features from composition fractions.

    Parameters
    ----------
    fracs : dict
        Element -> atomic fraction (e.g., {'Co': 0.2, 'Cr': 0.2, ...})

    Returns
    -------
    dict of features
    """
    active = {el: c for el, c in fracs.items() if c > 0}
    features = {}

    properties = {
        'r': RADII, 'mu': SHEAR_MOD, 'K': BULK_MOD, 'EN': EN,
        'Tm': TM, 'VEC': VEC_VALS, 'mass': MASS, 'a_fcc': A_FCC,
    }

    for prop_name, prop_dict in properties.items():
        vals = np.array([prop_dict[el] for el in ELEMENTS])
        cs = np.array([fracs.get(el, 0.0) for el in ELEMENTS])
        active_vals = [prop_dict[el] for el in active]

        mean_val = np.sum(cs * vals)
        features[f'{prop_name}_mean'] = mean_val

        var_val = np.sum(cs * (vals - mean_val)**2)
        features[f'{prop_name}_var'] = var_val

        if mean_val != 0:
            features[f'{prop_name}_delta'] = np.sqrt(var_val) / abs(mean_val)
        else:
            features[f'{prop_name}_delta'] = 0.0

        features[f'{prop_name}_range'] = max(active_vals) - min(active_vals)

    return features


def compute_hea_descriptors(fracs):
    """Compute HEA descriptors (delta, dS_mix, dH_mix, VEC, etc.).

    Parameters
    ----------
    fracs : dict
        Element -> atomic fraction

    Returns
    -------
    dict of descriptors
    """
    active = {el: c for el, c in fracs.items() if c > 0}
    n_comp = len(active)

    r_bar = sum(c * RADII[el] for el, c in fracs.items())
    delta = np.sqrt(sum(c * (1 - RADII[el] / r_bar)**2 for el, c in fracs.items()))

    dS_mix = -R_GAS * sum(c * np.log(c) for c in active.values())

    vec = sum(c * VEC_VALS[el] for el, c in fracs.items())

    en_bar = sum(c * EN[el] for el, c in fracs.items())
    delta_chi = np.sqrt(sum(c * (EN[el] - en_bar)**2 for el, c in fracs.items()))

    tm_bar = sum(c * TM[el] for el, c in fracs.items())
    mu_bar = sum(c * SHEAR_MOD[el] for el, c in fracs.items())
    delta_mu = np.sqrt(sum(c * (1 - SHEAR_MOD[el] / mu_bar)**2 for el, c in fracs.items()))

    dH_mix = 0.0
    for i, el_i in enumerate(ELEMENTS):
        for j, el_j in enumerate(ELEMENTS):
            if i < j:
                key = (el_i, el_j) if (el_i, el_j) in HMIX else (el_j, el_i)
                if key in HMIX:
                    dH_mix += 4 * HMIX[key] * fracs.get(el_i, 0) * fracs.get(el_j, 0)

    omega = tm_bar * dS_mix / (abs(dH_mix) * 1000) if abs(dH_mix) > 0.01 else np.inf

    # VLC Phi_VLC (sphere volumes — intentional)
    V = {el: (4 / 3) * np.pi * (RADII[el] * 1e-12)**3 for el in ELEMENTS}
    V_bar = sum(c * V[el] for el, c in fracs.items())
    a_bar = sum(c * A_FCC[el] for el, c in fracs.items())
    b_burg = a_bar / np.sqrt(2) * 1e-10
    sigma_dV2 = sum(c * (V[el] - V_bar)**2 for el, c in fracs.items())
    phi_vlc = sigma_dV2 / b_burg**6

    # Labusch combined misfit
    alpha_L = 16
    delta_r_i = {el: (RADII[el] - r_bar) / r_bar for el in ELEMENTS}
    delta_mu_i = {el: (SHEAR_MOD[el] - mu_bar) / mu_bar for el in ELEMENTS}
    eps_L = np.sqrt(
        sum(c * delta_mu_i[el]**2 for el, c in fracs.items())
        + alpha_L**2 * sum(c * delta_r_i[el]**2 for el, c in fracs.items())
    )

    # Toda-Caraballo 2015 (faithful — Gypen-Deruyttere superposition with α=16)
    # Z calibrated to Cantor σ_0 = 125 MPa (Otto 2013); held fixed thereafter.
    active = {el: c for el, c in fracs.items() if c > 0}
    alpha_TC = 16.0
    Z_TC = 1.0 / 642.0
    sumB_pow = 0.0
    for el, c in active.items():
        eta_i = 2 * (SHEAR_MOD[el] - mu_bar) / (SHEAR_MOD[el] + mu_bar)
        etap_i = eta_i / (1.0 + 0.5 * abs(eta_i))
        delta_i = (A_FCC[el] - a_bar) / a_bar
        eps_i = np.sqrt(etap_i**2 + (alpha_TC * delta_i)**2)
        B_i = 3.0 * (mu_bar * 1e9) * (eps_i ** (4/3)) * Z_TC
        sumB_pow += (B_i ** 1.5) * c
    sigma_TC = 3.06 * (sumB_pow ** (2/3)) / 1e6  # MPa

    return {
        'n_comp': n_comp, 'delta': delta, 'dS_mix': dS_mix,
        'VEC': vec, 'delta_chi': delta_chi, 'Tm_bar': tm_bar,
        'mu_bar': mu_bar, 'delta_mu': delta_mu,
        'dH_mix': dH_mix, 'Omega': omega,
        'Phi_VLC': phi_vlc, 'eps_Labusch': eps_L, 'a_bar': a_bar,
        'sigma_TC': sigma_TC,
    }


def compute_sisso_inputs(fracs, grain_size):
    """Compute the 3 SISSO Full features for prediction.

    SISSO equation: σ_y = c1*(r_var/r_range) + c2*(d^{-1/2}/dS_mix)
                        + c3*(EN_var/delta_mu) + intercept

    Returns dict with the 3 feature values.
    """
    oliynyk = compute_oliynyk_features(fracs)
    desc = compute_hea_descriptors(fracs)

    r_var = oliynyk['r_var']
    r_range = oliynyk['r_range']
    EN_var = oliynyk['EN_var']
    d_inv_sqrt = grain_size ** (-0.5)
    dS_mix = desc['dS_mix']
    delta_mu = desc['delta_mu']

    # Avoid division by zero
    feat1 = r_var / r_range if r_range > 0 else 0.0
    feat2 = d_inv_sqrt / dS_mix if dS_mix > 0 else 0.0
    feat3 = EN_var / delta_mu if delta_mu > 0 else 0.0

    return {
        'r_var/r_range': feat1,
        'd_inv_sqrt/dS_mix': feat2,
        'EN_var/delta_mu': feat3,
    }


def compute_sisso_robust_inputs(fracs, grain_size):
    """Compute the 3 SISSO Robust features for prediction.

    Robust equation (May 2026): σ_y = c1*(r_var/sigma_TC) + c2*(d^{-1/2}/dS_mix)
                                    + c3*(EN_var - Phi_VLC) + intercept
    Discovered by sisso_robust.py with δ_μ excluded from the SIS feature pool
    and the corrected Toda-Caraballo σ_TC included as a candidate descriptor.
    """
    oliynyk = compute_oliynyk_features(fracs)
    desc = compute_hea_descriptors(fracs)

    r_var = oliynyk['r_var']
    EN_var = oliynyk['EN_var']
    d_inv_sqrt = grain_size ** (-0.5)
    dS_mix = desc['dS_mix']
    phi_vlc = desc['Phi_VLC']
    sigma_TC = desc['sigma_TC']

    feat1 = r_var / sigma_TC if sigma_TC > 0 else 0.0
    feat2 = d_inv_sqrt / dS_mix if dS_mix > 0 else 0.0
    feat3 = EN_var - phi_vlc

    return {
        'r_var/sigma_TC': feat1,
        'd_inv_sqrt/dS_mix': feat2,
        'EN_var-Phi_VLC': feat3,
    }


# ============================================================
# SECTION 2: FORMULA PARSING
# ============================================================

def parse_formula(formula):
    """Parse alloy formula to normalized atomic fractions.

    Handles formats like:
      'Al0.25Co1Fe1Ni1' -> {'Al': 0.1, 'Co': 0.4, 'Fe': 0.4, 'Ni': 0.4} (normalized)
      'CoCrFeMnNi' -> equiatomic
      'Al0.3CoCrFeMnNi' -> Al=0.3, rest=1.0 each

    Returns
    -------
    dict or None
        Element -> fraction (normalized to sum=1), None if parse fails
    """
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)

    if not matches:
        return None

    raw = {}
    for elem, amt in matches:
        if not elem:
            continue
        raw[elem] = float(amt) if amt else 1.0

    total = sum(raw.values())
    if total <= 0:
        return None

    fracs = {el: raw.get(el, 0.0) / total for el in ELEMENTS}
    return fracs


def validate_composition(fracs):
    """Check all non-zero elements are in our system."""
    for el, c in fracs.items():
        if c > 0 and el not in ELEMENTS:
            return False
    # Check no unknown elements
    for el in fracs:
        if el not in ELEMENTS:
            return False
    return True


def parse_formula_flexible(formula):
    """Parse formula, returning fracs dict and list of all elements found."""
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)

    if not matches:
        return None, []

    raw = {}
    all_elements = []
    for elem, amt in matches:
        if not elem:
            continue
        all_elements.append(elem)
        raw[elem] = float(amt) if amt else 1.0

    total = sum(raw.values())
    if total <= 0:
        return None, all_elements

    # Check if all elements are in our system
    for el in all_elements:
        if el not in ELEMENTS:
            return None, all_elements

    fracs = {el: raw.get(el, 0.0) / total for el in ELEMENTS}
    return fracs, all_elements


# ============================================================
# SECTION 3: EXTERNAL DATA LOADING
# ============================================================

def load_citrine_mpea():
    """Load and filter the Citrine/Borg MPEA dataset from GitHub.

    Filters for:
      - FCC phase
      - Has yield strength
      - Has grain size
      - All elements ⊆ {Al, Co, Cr, Cu, Fe, Mn, Ni, V}
    """
    print("\n--- Loading Citrine/Borg MPEA Dataset ---")
    url = 'https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv'

    try:
        df = pd.read_csv(url)
        print(f"  Downloaded {len(df)} entries")
    except Exception as e:
        print(f"  WARNING: Could not download Citrine data: {e}")
        print("  Attempting local fallback...")
        local_path = f'{DATA_DIR}/citrine_mpea_dataset.csv'
        try:
            df = pd.read_csv(local_path)
            print(f"  Loaded {len(df)} entries from local cache")
        except FileNotFoundError:
            print("  No local cache found. Skipping Citrine data.")
            return pd.DataFrame()

    # Identify relevant columns (handle LaTeX-formatted names)
    phase_col = [c for c in df.columns if 'BCC' in c or 'FCC' in c or 'phase' in c.lower()]
    ys_col = [c for c in df.columns if 'YS' in c and 'MPa' in c]
    gs_col = [c for c in df.columns if 'grain' in c.lower() and ('m)' in c or 'µm' in c or 'um' in c.lower())]
    formula_col = [c for c in df.columns if 'formula' in c.lower() or 'FORMULA' in c]

    if not formula_col:
        formula_col = [c for c in df.columns if 'composition' in c.lower()]

    print(f"  Phase columns: {phase_col}")
    print(f"  YS columns: {ys_col}")
    print(f"  GS columns: {gs_col}")
    print(f"  Formula columns: {formula_col}")

    if not phase_col or not ys_col or not gs_col or not formula_col:
        print("  Could not identify required columns. Available:")
        for c in df.columns:
            print(f"    '{c}'")
        return pd.DataFrame()

    pc = phase_col[0]
    yc = ys_col[0]
    gc = gs_col[0]
    fc = formula_col[0]

    # Filter for FCC
    df_fcc = df[df[pc].astype(str).str.contains('FCC', case=False, na=False)].copy()
    print(f"  FCC entries: {len(df_fcc)}")

    # Filter for valid YS and GS
    df_fcc[yc] = pd.to_numeric(df_fcc[yc], errors='coerce')
    df_fcc[gc] = pd.to_numeric(df_fcc[gc], errors='coerce')
    df_valid = df_fcc.dropna(subset=[yc, gc]).copy()
    df_valid = df_valid[(df_valid[yc] > 0) & (df_valid[gc] > 0)]
    print(f"  With valid YS + GS: {len(df_valid)}")

    # Parse compositions and filter for our element system
    records = []
    skipped_elements = set()
    for _, row in df_valid.iterrows():
        formula = str(row[fc])
        fracs, all_elems = parse_formula_flexible(formula)
        if fracs is None:
            for el in all_elems:
                if el not in ELEMENTS:
                    skipped_elements.add(el)
            continue

        record = {
            'source': 'Citrine',
            'alloy': formula,
            'GrainSize': row[gc],
            'YS_exp': row[yc],
            'is_hv_converted': False,
            'test_mode': 'mixed',
            'data_quality': 'aggregated',
        }
        for el in ELEMENTS:
            record[f'{el}_frac'] = fracs[el]
        records.append(record)

    if skipped_elements:
        print(f"  Skipped elements outside our system: {skipped_elements}")
    print(f"  Valid entries in our element system: {len(records)}")

    # Quality filtering
    df_cit = pd.DataFrame(records)
    if df_cit.empty:
        return df_cit

    n_before = len(df_cit)

    # Remove sub-micron grains (nano-regime, HP breaks down)
    df_cit = df_cit[df_cit['GrainSize'] >= 3.0]
    n_nano = n_before - len(df_cit)

    # Remove grains > 500 um (far outside training range, unreliable)
    n_before2 = len(df_cit)
    df_cit = df_cit[df_cit['GrainSize'] <= 500]
    n_large = n_before2 - len(df_cit)

    # Remove YS > 800 MPa (likely precipitate-hardened, nano-twinned, or SPD)
    n_before3 = len(df_cit)
    df_cit = df_cit[df_cit['YS_exp'] <= 800]
    n_high_ys = n_before3 - len(df_cit)

    print(f"  Quality filtering:")
    print(f"    Removed {n_nano} sub-micron (< 3 μm, nano-regime)")
    print(f"    Removed {n_large} oversized (> 500 μm)")
    print(f"    Removed {n_high_ys} high-YS (> 800 MPa, likely non-single-phase)")
    print(f"    Remaining: {len(df_cit)} entries")

    return df_cit


def load_schneider_crfeni():
    """CrFeNi benchmark — VERIFIED data from open-access publication.

    Source: Schneider & Laplanche, Data in Brief 34 (2021) 106717, Table 11.
            PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7807211/
    Test mode: COMPRESSION (sigma_0.2%)
    Temperature: 293 K
    """
    print("\n--- Loading Schneider CrFeNi (VERIFIED, compression) ---")

    # Exact values from Table 11 of the open-access Data in Brief paper
    # HP params from paper: sigma_0 = 80 ± 8 MPa, k_y = 966 ± 25 MPa·um^{1/2}
    crfeni_data = [
        # (grain_size_um, YS_MPa)
        (10.0, 359),   # ± (1, 2)
        (19.0, 286),   # ± (2, 9)
        (34.0, 261),   # ± (1, 9)
        (55.0, 213),   # ± (2, 2)
        (75.0, 185),   # ± (4, 3)
        (160.0, 149),  # ± (8, 7)
        # Note: 327 ± 20 um, 163 ± 8 MPa EXCLUDED (anomalous — above HP trend)
    ]

    records = []
    for gs, ys in crfeni_data:
        records.append({
            'source': 'Schneider2021',
            'alloy': 'CrFeNi',
            'GrainSize': gs,
            'YS_exp': ys,
            'is_hv_converted': False,
            'test_mode': 'compression',
            'data_quality': 'verified',
            'Al_frac': 0, 'Co_frac': 0, 'Cr_frac': 1/3, 'Cu_frac': 0,
            'Fe_frac': 1/3, 'Mn_frac': 0, 'Ni_frac': 1/3, 'V_frac': 0,
        })

    print(f"  Loaded {len(records)} CrFeNi entries (compression, verified from Table 11)")
    print(f"  HP params: sigma_0 = 80 MPa, k_y = 966 MPa·um^(1/2)")
    return pd.DataFrame(records)


def load_otto_cantor():
    """CoCrFeMnNi Cantor alloy — from published HP parameters.

    Source: Otto et al., Acta Mater. 61 (2013) 5743-5755.
    HP params (well-established, confirmed by multiple citing papers):
        sigma_0 = 125 MPa, k_HP = 494 MPa·um^{1/2}
    Grain sizes: 4.4, 50, 155 um (recrystallized at 1073, 1273, 1423 K)
    Test mode: TENSION
    Temperature: 293 K

    YS computed from HP fit (exact table values behind paywall,
    but HP params are multiply-confirmed in literature).
    """
    print("\n--- Loading Otto et al. 2013 (Cantor Alloy, HP-derived) ---")

    sigma_0 = 125.0  # MPa
    k_HP = 494.0     # MPa·um^{1/2}

    otto_gs = [4.4, 50.0, 155.0]
    records = []
    for gs in otto_gs:
        ys = sigma_0 + k_HP * gs**(-0.5)
        records.append({
            'source': 'Otto2013',
            'alloy': 'CoCrFeMnNi',
            'GrainSize': gs,
            'YS_exp': round(ys, 0),
            'is_hv_converted': False,
            'test_mode': 'tension',
            'data_quality': 'hp_derived',
            'Al_frac': 0, 'Co_frac': 0.2, 'Cr_frac': 0.2, 'Cu_frac': 0,
            'Fe_frac': 0.2, 'Mn_frac': 0.2, 'Ni_frac': 0.2, 'V_frac': 0,
        })

    print(f"  Loaded {len(records)} entries (tension, HP-derived)")
    print(f"  HP params: sigma_0 = {sigma_0:.0f} MPa, k_HP = {k_HP:.0f} MPa·um^(1/2)")
    for r in records:
        print(f"    d = {r['GrainSize']:.1f} um -> YS = {r['YS_exp']:.0f} MPa")
    return pd.DataFrame(records)


def load_huang_2019_hv():
    """Huang et al. 2019 — VERIFIED HV vs grain size for multiple FCC alloys.

    Source: Huang, Su, Wu & Lin, Entropy 21(3) (2019) 297.
            PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7514778/
    Test mode: Vickers hardness (indentation)
    HV converted to YS via C_eff = 5.13 (from our hardness analysis)
    Compositions in our element system: CoCrFeMnNi, CoCrNi, CoCrFeNi, CoNiMn

    NOTE: FeCoNiCrPd EXCLUDED (Pd outside our element system)
    NOTE: FeCoNiMn included as it only contains elements in our system
    """
    print("\n--- Loading Huang et al. 2019 (VERIFIED HV data) ---")

    records = []

    # All data from Table 1 of Huang et al. 2019 (verified from PMC full text)
    hv_datasets = {
        'CoCrFeMnNi': {
            'fracs': {'Co': 0.2, 'Cr': 0.2, 'Fe': 0.2, 'Mn': 0.2, 'Ni': 0.2},
            'data': [(3.7, 176.6), (13.9, 147.9), (63.3, 136.1),
                     (120.2, 132.8), (209.6, 128.7)],
            'H0': 122.3, 'KH': 103.1,
        },
        'CoCrNi': {
            'fracs': {'Co': 1/3, 'Cr': 1/3, 'Ni': 1/3},
            'data': [(4.0, 255.1), (12.3, 195.5), (69.3, 158.0),
                     (101.7, 153.3), (152.6, 151.5)],
            'H0': 128.7, 'KH': 248.7,
        },
        'CoCrFeNi': {
            'fracs': {'Co': 0.25, 'Cr': 0.25, 'Fe': 0.25, 'Ni': 0.25},
            'data': [(4.2, 185.6), (13.1, 154.9), (64.2, 133.6),
                     (108.1, 129.1), (153.9, 125.4)],
            'H0': 114.7, 'KH': 145.5,
        },
        'CoNiMn': {
            'fracs': {'Co': 1/3, 'Mn': 1/3, 'Ni': 1/3},
            'data': [(17.2, 157.8), (23.8, 150.9), (41.4, 145.1),
                     (83.9, 139.8), (166.4, 136.5)],
            'H0': 126.0, 'KH': 126.8,
        },
        'CoFeNiMn': {
            'fracs': {'Co': 0.25, 'Fe': 0.25, 'Mn': 0.25, 'Ni': 0.25},
            'data': [(9.8, 144.5), (27.5, 134.4), (55.9, 127.6),
                     (87.0, 121.9), (208.3, 118.9)],
            'H0': 112.4, 'KH': 104.1,
        },
    }

    for alloy_name, info in hv_datasets.items():
        fracs = info['fracs']
        for gs, hv in info['data']:
            ys = hv * 9.807 / C_EFF  # Convert HV to YS (MPa)
            record = {
                'source': 'Huang2019',
                'alloy': alloy_name,
                'GrainSize': gs,
                'YS_exp': ys,
                'HV_exp': hv,
                'is_hv_converted': True,
                'test_mode': 'hardness',
                'data_quality': 'verified_hv',
            }
            for el in ELEMENTS:
                record[f'{el}_frac'] = fracs.get(el, 0.0)
            records.append(record)

    print(f"  Loaded {len(records)} entries from {len(hv_datasets)} alloys")
    print(f"  HV -> YS conversion: C_eff = {C_EFF}")
    for name, info in hv_datasets.items():
        print(f"    {name}: {len(info['data'])} points, H0={info['H0']}, KH={info['KH']}")

    return pd.DataFrame(records)


def ceff_sensitivity_analysis(df_huang_base, models):
    """Test sensitivity of external validation R² to C_eff conversion factor.

    Re-converts Huang HV data using a range of C_eff values and reports
    how the M3 and SISSO external validation R² changes.

    Parameters
    ----------
    df_huang_base : DataFrame
        Huang data with HV_exp column (unconverted HV values).
    models : dict
        Fitted model objects from fit_training_models().

    Returns
    -------
    DataFrame with sensitivity results.
    """
    print("\n" + "=" * 70)
    print("C_EFF SENSITIVITY ANALYSIS (Huang HV -> YS conversion)")
    print("=" * 70)

    c_eff_values = [3.0, 4.0, 5.13, 6.0, 7.0]
    sensitivity_rows = []

    for c_val in c_eff_values:
        df_test = df_huang_base.copy()
        df_test['YS_exp'] = df_test['HV_exp'] * 9.807 / c_val

        # Generate predictions
        sisso_preds = []
        sisso_robust_preds = []
        m3_preds = []
        for _, row in df_test.iterrows():
            fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
            gs = row['GrainSize']
            sisso_preds.append(predict_sisso(fracs, gs, models['sisso_model']))
            sisso_robust_preds.append(predict_sisso_robust(fracs, gs, models['sisso_robust_model']))
            m3_preds.append(predict_m3(fracs, gs, models['m3_model']))

        df_test['YS_SISSO'] = sisso_preds
        df_test['YS_SISSO_robust'] = sisso_robust_preds
        df_test['YS_M3'] = m3_preds

        for model_name, pred_col in [('SISSO', 'YS_SISSO'),
                                      ('SISSO_Robust', 'YS_SISSO_robust'),
                                      ('M3', 'YS_M3')]:
            r2 = r2_score(df_test['YS_exp'], df_test[pred_col])
            rmse = np.sqrt(mean_squared_error(df_test['YS_exp'], df_test[pred_col]))
            mae = mean_absolute_error(df_test['YS_exp'], df_test[pred_col])
            sensitivity_rows.append({
                'C_eff': c_val,
                'Model': model_name,
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'n': len(df_test),
            })

    df_sens = pd.DataFrame(sensitivity_rows)

    # Print table
    print(f"\n  {'C_eff':>6s} {'Model':<14s} {'n':>4s} {'R²':>8s} {'RMSE':>8s} {'MAE':>8s}")
    print("  " + "-" * 48)
    for _, row in df_sens.iterrows():
        marker = " <--" if row['C_eff'] == C_EFF else ""
        print(f"  {row['C_eff']:6.2f} {row['Model']:<14s} {row['n']:4d} "
              f"{row['R2']:8.3f} {row['RMSE']:8.1f} {row['MAE']:8.1f}{marker}")

    print(f"\n  NOTE: C_eff = {C_EFF} is our estimate from hardness analysis.")
    print(f"  Tabor's standard value is 3.0. C_eff > 3 reflects work-hardening under indenter.")

    return df_sens


def remove_training_overlaps(df_ext, df_train, tol_gs=5.0, tol_comp=0.02):
    """Remove external entries that overlap with training data.

    Overlap = same composition (within tol_comp in each element)
              AND similar grain size (within tol_gs μm).
    """
    if df_ext.empty:
        return df_ext

    frac_cols = [f'{el}_frac' for el in ELEMENTS]
    n_before = len(df_ext)
    mask = np.ones(len(df_ext), dtype=bool)

    for i, row in df_ext.iterrows():
        for _, trow in df_train.iterrows():
            comp_match = all(
                abs(row[f'{el}_frac'] - trow[f'{el}_frac']) < tol_comp
                for el in ELEMENTS
            )
            gs_match = abs(row['GrainSize'] - trow['GrainSize']) < tol_gs
            if comp_match and gs_match:
                mask[df_ext.index == i] = False
                break

    df_clean = df_ext[mask].copy()
    n_removed = n_before - len(df_clean)
    if n_removed > 0:
        print(f"  Removed {n_removed} entries overlapping with training data")
    return df_clean


def load_all_external_data(df_train):
    """Load and combine all external data sources."""
    print("\n" + "=" * 70)
    print("LOADING EXTERNAL DATA")
    print("=" * 70)

    dfs = []

    # 1. Citrine (aggregated, quality-filtered)
    df_citrine = load_citrine_mpea()
    if not df_citrine.empty:
        df_citrine = remove_training_overlaps(df_citrine, df_train)
        dfs.append(df_citrine)

    # 2. CrFeNi (Schneider, VERIFIED from open-access paper)
    df_crfeni = load_schneider_crfeni()
    df_crfeni = remove_training_overlaps(df_crfeni, df_train)
    dfs.append(df_crfeni)

    # 3. CoCrFeMnNi (Otto 2013, HP-derived from published parameters)
    df_otto = load_otto_cantor()
    df_otto = remove_training_overlaps(df_otto, df_train)
    dfs.append(df_otto)

    # 4. HV data (Huang 2019, VERIFIED from open-access paper)
    df_huang = load_huang_2019_hv()
    df_huang = remove_training_overlaps(df_huang, df_train)
    dfs.append(df_huang)

    df_ext = pd.concat(dfs, ignore_index=True)

    # Ensure all frac columns exist
    for el in ELEMENTS:
        col = f'{el}_frac'
        if col not in df_ext.columns:
            df_ext[col] = 0.0
        df_ext[col] = df_ext[col].fillna(0.0)

    # Fill optional columns
    for col in ['test_mode', 'data_quality', 'HV_exp']:
        if col not in df_ext.columns:
            df_ext[col] = np.nan

    print(f"\n  Total external data points: {len(df_ext)}")
    print(f"  By source:")
    for src in df_ext['source'].unique():
        n = (df_ext['source'] == src).sum()
        quality = df_ext.loc[df_ext['source'] == src, 'data_quality'].iloc[0]
        print(f"    {src}: {n}  (quality: {quality})")
    print(f"  HV-converted: {df_ext['is_hv_converted'].sum()}")
    print(f"  Tension: {(df_ext.get('test_mode') == 'tension').sum()}")
    print(f"  Compression: {(df_ext.get('test_mode') == 'compression').sum()}")

    return df_ext


# ============================================================
# SECTION 4: MODEL FITTING ON TRAINING DATA
# ============================================================

def fit_training_models():
    """Fit SISSO and M3 on the training data, return model objects."""
    print("\n" + "=" * 70)
    print("FITTING MODELS ON TRAINING DATA")
    print("=" * 70)

    # Load training data
    df = pd.read_csv(DATA_CSV)
    if 'eps_Labusch.1' in df.columns:
        df = df.drop(columns=['eps_Labusch.1'])
    df = df.replace([np.inf, -np.inf], np.nan)

    df_ys = df.dropna(subset=['YS']).copy()
    y = df_ys['YS'].values
    n = len(y)
    d = df_ys['GrainSize'].values
    d_inv_sqrt = d ** (-0.5)
    print(f"  Training data: {n} alloys, YS range {y.min():.0f}-{y.max():.0f} MPa")

    # --- Fit SISSO: reconstruct features and refit OLS ---
    print("\n  Fitting SISSO Full (OLS on 3 symbolic terms)...")

    # Compute SISSO features for training data
    sisso_feats = np.zeros((n, 3))
    for i, (_, row) in enumerate(df_ys.iterrows()):
        fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
        sf = compute_sisso_inputs(fracs, row['GrainSize'])
        sisso_feats[i, 0] = sf['r_var/r_range']
        sisso_feats[i, 1] = sf['d_inv_sqrt/dS_mix']
        sisso_feats[i, 2] = sf['EN_var/delta_mu']

    # Refit OLS to get exact coefficients
    reg_sisso = LinearRegression().fit(sisso_feats, y)
    sisso_coefs = reg_sisso.coef_
    sisso_intercept = reg_sisso.intercept_
    sisso_train_pred = reg_sisso.predict(sisso_feats)
    sisso_train_r2 = r2_score(y, sisso_train_pred)

    print(f"    Coefficients: {sisso_coefs}")
    print(f"    Intercept: {sisso_intercept:.4f}")
    print(f"    Train R² = {sisso_train_r2:.4f}")

    # --- Fit SISSO Robust: same structure, avoids EN_var/delta_mu singularity ---
    print("\n  Fitting SISSO Robust (OLS on 3 robust terms)...")

    sisso_robust_feats = np.zeros((n, 3))
    for i, (_, row) in enumerate(df_ys.iterrows()):
        fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
        sf = compute_sisso_robust_inputs(fracs, row['GrainSize'])
        sisso_robust_feats[i, 0] = sf['r_var/sigma_TC']
        sisso_robust_feats[i, 1] = sf['d_inv_sqrt/dS_mix']
        sisso_robust_feats[i, 2] = sf['EN_var-Phi_VLC']

    reg_sisso_robust = LinearRegression().fit(sisso_robust_feats, y)
    robust_coefs = reg_sisso_robust.coef_
    robust_intercept = reg_sisso_robust.intercept_
    robust_train_pred = reg_sisso_robust.predict(sisso_robust_feats)
    robust_train_r2 = r2_score(y, robust_train_pred)

    print(f"    Coefficients: {robust_coefs}")
    print(f"    Intercept: {robust_intercept:.4f}")
    print(f"    Train R² = {robust_train_r2:.4f}")

    # --- Fit M3: σ₀(7 elem, drop Ni as reference) + k·d^{-1/2} ---
    print("\n  Fitting M3: composition-dependent Hall-Petch (7 elem, Ni=ref)...")
    elem_frac_cols_7 = [f'{el}_frac' for el in ELEMENTS if el != 'Ni']
    X_m3 = np.column_stack([np.ones(n), df_ys[elem_frac_cols_7].values, d_inv_sqrt])
    reg_m3 = LinearRegression(fit_intercept=False).fit(X_m3, y)
    m3_train_pred = reg_m3.predict(X_m3)
    m3_train_r2 = r2_score(y, m3_train_pred)
    print(f"    M3 coefficients (Ni = reference element):")
    print(f"      intercept={reg_m3.coef_[0]:.1f}")
    for i, el in enumerate(el for el in ELEMENTS if el != 'Ni'):
        print(f"      {el}: {reg_m3.coef_[i+1]:.1f}")
    print(f"      k_HP: {reg_m3.coef_[-1]:.1f}")
    print(f"    Train R² = {m3_train_r2:.4f}")
    print(f"    Parameters: k_m3 = 9 (intercept + 7 elem + k_HP)")

    return {
        'sisso_model': reg_sisso,
        'sisso_coefs': sisso_coefs,
        'sisso_intercept': sisso_intercept,
        'sisso_robust_model': reg_sisso_robust,
        'sisso_robust_coefs': robust_coefs,
        'sisso_robust_intercept': robust_intercept,
        'm3_model': reg_m3,
        'df_train': df_ys,
        'y_train': y,
    }


# ============================================================
# SECTION 5: PREDICTION PIPELINE
# ============================================================

def predict_sisso(fracs, grain_size, sisso_model):
    """Predict YS using SISSO Full model."""
    sf = compute_sisso_inputs(fracs, grain_size)
    X = np.array([[sf['r_var/r_range'], sf['d_inv_sqrt/dS_mix'], sf['EN_var/delta_mu']]])
    return sisso_model.predict(X)[0]


def predict_sisso_robust(fracs, grain_size, sisso_robust_model):
    """Predict YS using SISSO Robust model (avoids EN_var/delta_mu singularity)."""
    sf = compute_sisso_robust_inputs(fracs, grain_size)
    X = np.array([[sf['r_var/sigma_TC'], sf['d_inv_sqrt/dS_mix'], sf['EN_var-Phi_VLC']]])
    return sisso_robust_model.predict(X)[0]


def predict_m3(fracs, grain_size, m3_model):
    """Predict YS using M3 composition-dependent HP model (7 elem, Ni=ref)."""
    d_inv_sqrt = grain_size ** (-0.5)
    elem_vals = [fracs.get(el, 0.0) for el in ELEMENTS if el != 'Ni']
    X = np.array([[1.0] + elem_vals + [d_inv_sqrt]])
    return m3_model.predict(X)[0]


def predict_all_external(df_ext, models):
    """Generate predictions for all external data points."""
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    sisso_preds = []
    sisso_robust_preds = []
    m3_preds = []

    for _, row in df_ext.iterrows():
        fracs = {el: row[f'{el}_frac'] for el in ELEMENTS}
        gs = row['GrainSize']

        sisso_preds.append(predict_sisso(fracs, gs, models['sisso_model']))
        sisso_robust_preds.append(predict_sisso_robust(fracs, gs, models['sisso_robust_model']))
        m3_preds.append(predict_m3(fracs, gs, models['m3_model']))

    df_ext = df_ext.copy()
    df_ext['YS_SISSO'] = sisso_preds
    df_ext['YS_SISSO_robust'] = sisso_robust_preds
    df_ext['YS_M3'] = m3_preds
    df_ext['residual_SISSO'] = df_ext['YS_exp'] - df_ext['YS_SISSO']
    df_ext['residual_SISSO_robust'] = df_ext['YS_exp'] - df_ext['YS_SISSO_robust']
    df_ext['residual_M3'] = df_ext['YS_exp'] - df_ext['YS_M3']

    print(f"  Generated predictions for {len(df_ext)} external data points")
    return df_ext


# ============================================================
# SECTION 6: ERROR METRICS & HP SLOPE ANALYSIS
# ============================================================

def compute_error_metrics(y_true, y_pred, label=""):
    """Compute R², RMSE, MAE, bias."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {'label': label, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'bias': bias, 'n': len(y_true)}


def evaluate_models(df_ext):
    """Evaluate SISSO and M3 per source and overall."""
    print("\n" + "=" * 70)
    print("ERROR METRICS")
    print("=" * 70)

    results = []

    MODEL_COLS = [('SISSO', 'YS_SISSO'), ('SISSO Robust', 'YS_SISSO_robust'), ('M3', 'YS_M3')]

    # Overall
    for model_name, pred_col in MODEL_COLS:
        metrics = compute_error_metrics(df_ext['YS_exp'], df_ext[pred_col], f'{model_name} (all)')
        results.append(metrics)

    # Per source
    for src in df_ext['source'].unique():
        mask = df_ext['source'] == src
        if mask.sum() < 2:
            continue
        for model_name, pred_col in MODEL_COLS:
            metrics = compute_error_metrics(
                df_ext.loc[mask, 'YS_exp'],
                df_ext.loc[mask, pred_col],
                f'{model_name} ({src})'
            )
            results.append(metrics)

    # Print table
    print(f"\n  {'Model':<25s} {'n':>4s} {'R²':>8s} {'RMSE':>8s} {'MAE':>8s} {'Bias':>8s}")
    print("  " + "-" * 61)
    for r in results:
        print(f"  {r['label']:<25s} {r['n']:4d} {r['R2']:8.3f} {r['RMSE']:8.1f} {r['MAE']:8.1f} {r['bias']:8.1f}")

    # --- Data Quality Tier Separation ---
    # Tier 1: Directly measured YS (Schneider compression + Citrine after overlap removal)
    # Tier 2: HP-derived data (Otto CoCrFeMnNi)
    # Tier 3: HV-converted data (Huang alloys)
    tier_defs = {
        'Tier 1 (measured YS)': df_ext['source'].isin(['Schneider2021', 'Citrine']),
        'Tier 2 (HP-derived)': df_ext['source'] == 'Otto2013',
        'Tier 3 (HV-converted)': df_ext['source'] == 'Huang2019',
    }

    tier_results = []
    print(f"\n  --- Results by Data Quality Tier ---")
    print(f"  {'Tier / Model':<35s} {'n':>4s} {'R²':>8s} {'RMSE':>8s} {'MAE':>8s} {'Bias':>8s}")
    print("  " + "-" * 69)

    for tier_name, tier_mask in tier_defs.items():
        df_tier = df_ext[tier_mask]
        if len(df_tier) < 2:
            print(f"  {tier_name}: insufficient data (n={len(df_tier)})")
            continue
        print(f"  {tier_name} (n={len(df_tier)}):")
        for model_name, pred_col in MODEL_COLS:
            m = compute_error_metrics(
                df_tier['YS_exp'], df_tier[pred_col],
                f'{tier_name} / {model_name}'
            )
            tier_results.append(m)
            print(f"    {model_name:<29s} {m['n']:4d} {m['R2']:8.3f} "
                  f"{m['RMSE']:8.1f} {m['MAE']:8.1f} {m['bias']:8.1f}")

    # Aggregate (repeat for completeness in tier table)
    print(f"  Aggregate (n={len(df_ext)}):")
    for model_name, pred_col in MODEL_COLS:
        m = compute_error_metrics(
            df_ext['YS_exp'], df_ext[pred_col],
            f'Aggregate / {model_name}'
        )
        tier_results.append(m)
        print(f"    {model_name:<29s} {m['n']:4d} {m['R2']:8.3f} "
              f"{m['RMSE']:8.1f} {m['MAE']:8.1f} {m['bias']:8.1f}")

    return results, tier_results


def hp_slope_analysis(df_ext):
    """Compare HP slopes for single-composition datasets.

    For compositions with multiple grain sizes, fit:
      YS = σ₀ + k_HP · d^{-1/2}
    and compare experimental vs predicted slopes.
    """
    print("\n" + "=" * 70)
    print("HALL-PETCH SLOPE ANALYSIS")
    print("=" * 70)

    slope_results = []

    # Group by alloy
    for alloy in df_ext['alloy'].unique():
        mask = df_ext['alloy'] == alloy
        df_alloy = df_ext[mask]

        if len(df_alloy) < 3:
            continue

        d_inv_sqrt = df_alloy['GrainSize'].values ** (-0.5)
        X = d_inv_sqrt.reshape(-1, 1)

        # Experimental slope
        reg_exp = LinearRegression().fit(X, df_alloy['YS_exp'].values)
        k_exp = reg_exp.coef_[0]
        sigma0_exp = reg_exp.intercept_

        # SISSO slope
        reg_sisso = LinearRegression().fit(X, df_alloy['YS_SISSO'].values)
        k_sisso = reg_sisso.coef_[0]
        sigma0_sisso = reg_sisso.intercept_

        # SISSO Robust slope
        reg_robust = LinearRegression().fit(X, df_alloy['YS_SISSO_robust'].values)
        k_robust = reg_robust.coef_[0]
        sigma0_robust = reg_robust.intercept_

        # M3 slope
        reg_m3 = LinearRegression().fit(X, df_alloy['YS_M3'].values)
        k_m3 = reg_m3.coef_[0]
        sigma0_m3 = reg_m3.intercept_

        slope_results.append({
            'alloy': alloy,
            'n_points': len(df_alloy),
            'source': df_alloy['source'].iloc[0],
            'k_HP_exp': k_exp,
            'sigma0_exp': sigma0_exp,
            'k_HP_SISSO': k_sisso,
            'sigma0_SISSO': sigma0_sisso,
            'k_HP_SISSO_robust': k_robust,
            'sigma0_SISSO_robust': sigma0_robust,
            'k_HP_M3': k_m3,
            'sigma0_M3': sigma0_m3,
        })

        print(f"\n  {alloy} ({len(df_alloy)} grain sizes, source: {df_alloy['source'].iloc[0]})")
        print(f"    Experimental:    σ₀ = {sigma0_exp:.0f} MPa,  k_HP = {k_exp:.0f} MPa·µm^(1/2)")
        print(f"    SISSO:           σ₀ = {sigma0_sisso:.0f} MPa,  k_HP = {k_sisso:.0f} MPa·µm^(1/2)")
        print(f"    SISSO Robust:    σ₀ = {sigma0_robust:.0f} MPa,  k_HP = {k_robust:.0f} MPa·µm^(1/2)")
        print(f"    M3:              σ₀ = {sigma0_m3:.0f} MPa,  k_HP = {k_m3:.0f} MPa·µm^(1/2)")

    return pd.DataFrame(slope_results)


# ============================================================
# SECTION 7: VISUALIZATION
# ============================================================

def plot_parity(df_ext, models, plot_num=67):
    """Parity plot: predicted vs measured YS, colored by source."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (model_name, pred_col) in zip(axes, [
            ('SISSO Full', 'YS_SISSO'), ('SISSO Robust', 'YS_SISSO_robust'), ('M3', 'YS_M3')]):
        # Training LOO for reference
        df_train = models['df_train']
        y_train = models['y_train']

        for src in df_ext['source'].unique():
            mask = df_ext['source'] == src
            ax.scatter(df_ext.loc[mask, 'YS_exp'], df_ext.loc[mask, pred_col],
                       c=SOURCE_COLORS.get(src, '#333333'), label=src,
                       s=60, alpha=0.8, edgecolors='k', linewidths=0.5, zorder=5)

        # HV-converted entries marked with x
        hv_mask = df_ext['is_hv_converted']
        if hv_mask.any():
            ax.scatter(df_ext.loc[hv_mask, 'YS_exp'], df_ext.loc[hv_mask, pred_col],
                       marker='x', c='red', s=40, zorder=6, label='HV-converted')

        # Perfect prediction line
        all_vals = np.concatenate([df_ext['YS_exp'].values, df_ext[pred_col].values])
        lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y=x')
        ax.plot([lo, hi], [lo + 50, hi + 50], 'k:', alpha=0.3)
        ax.plot([lo, hi], [lo - 50, hi - 50], 'k:', alpha=0.3)

        # Metrics
        r2 = r2_score(df_ext['YS_exp'], df_ext[pred_col])
        rmse = np.sqrt(mean_squared_error(df_ext['YS_exp'], df_ext[pred_col]))
        ax.text(0.05, 0.92, f'R² = {r2:.3f}\nRMSE = {rmse:.1f} MPa',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Experimental YS (MPa)', fontsize=12)
        ax.set_ylabel('Predicted YS (MPa)', fontsize=12)
        ax.set_title(f'{model_name} — External Validation', fontsize=13)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')

    plt.tight_layout()
    path = f'{PLOT_DIR}/{plot_num}_external_parity.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_hp_slopes(df_ext, df_slopes, plot_num=68):
    """Hall-Petch slope comparison: YS vs d^{-1/2}, exp vs predicted lines."""
    alloys_to_plot = df_slopes['alloy'].tolist()
    n_alloys = len(alloys_to_plot)
    if n_alloys == 0:
        print("  No alloys with ≥3 grain sizes for HP slope plot")
        return

    fig, axes = plt.subplots(1, n_alloys, figsize=(5 * n_alloys, 5), squeeze=False)
    axes = axes[0]

    for i, alloy in enumerate(alloys_to_plot):
        ax = axes[i]
        mask = df_ext['alloy'] == alloy
        df_a = df_ext[mask].sort_values('GrainSize')

        d_inv_sqrt = df_a['GrainSize'].values ** (-0.5)
        d_range = np.linspace(d_inv_sqrt.min() * 0.8, d_inv_sqrt.max() * 1.2, 50)

        # Experimental
        ax.scatter(d_inv_sqrt, df_a['YS_exp'], c='k', s=80, zorder=5, label='Exp')
        reg_exp = LinearRegression().fit(d_inv_sqrt.reshape(-1, 1), df_a['YS_exp'].values)
        ax.plot(d_range, reg_exp.predict(d_range.reshape(-1, 1)), 'k-', lw=2)

        # SISSO
        ax.scatter(d_inv_sqrt, df_a['YS_SISSO'], c='#1f77b4', s=60, marker='s', zorder=4, label='SISSO Full')
        reg_s = LinearRegression().fit(d_inv_sqrt.reshape(-1, 1), df_a['YS_SISSO'].values)
        ax.plot(d_range, reg_s.predict(d_range.reshape(-1, 1)), '#1f77b4', ls='--', lw=1.5)

        # SISSO Robust
        ax.scatter(d_inv_sqrt, df_a['YS_SISSO_robust'], c='#2ca02c', s=60, marker='D', zorder=4, label='SISSO Robust')
        reg_sr = LinearRegression().fit(d_inv_sqrt.reshape(-1, 1), df_a['YS_SISSO_robust'].values)
        ax.plot(d_range, reg_sr.predict(d_range.reshape(-1, 1)), '#2ca02c', ls='--', lw=1.5)

        # M3
        ax.scatter(d_inv_sqrt, df_a['YS_M3'], c='#ff7f0e', s=60, marker='^', zorder=4, label='M3')
        reg_m = LinearRegression().fit(d_inv_sqrt.reshape(-1, 1), df_a['YS_M3'].values)
        ax.plot(d_range, reg_m.predict(d_range.reshape(-1, 1)), '#ff7f0e', ls='--', lw=1.5)

        # Slope annotations
        row = df_slopes[df_slopes['alloy'] == alloy].iloc[0]
        ax.text(0.05, 0.05,
                f"k_HP:\n  Exp: {row['k_HP_exp']:.0f}\n  SISSO: {row['k_HP_SISSO']:.0f}"
                f"\n  Robust: {row['k_HP_SISSO_robust']:.0f}\n  M3: {row['k_HP_M3']:.0f}",
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_xlabel(r'$d^{-1/2}$ ($\mu m^{-1/2}$)', fontsize=12)
        ax.set_ylabel('Yield Strength (MPa)', fontsize=12)
        ax.set_title(alloy, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = f'{PLOT_DIR}/{plot_num}_external_hp_slopes.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_error_bars(error_results, plot_num=69):
    """Bar chart of error metrics by source and model."""
    df_err = pd.DataFrame(error_results)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, ['R2', 'RMSE', 'MAE']):
        bars = df_err[['label', metric]].copy()
        colors = []
        for label in bars['label']:
            if 'SISSO Robust' in label:
                colors.append('#2ca02c')
            elif 'SISSO' in label:
                colors.append('#1f77b4')
            else:
                colors.append('#ff7f0e')

        ax.barh(range(len(bars)), bars[metric].values, color=colors, alpha=0.8,
                edgecolor='k', linewidth=0.5)
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels(bars['label'], fontsize=9)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=13, fontweight='bold')

        if metric == 'R2':
            ax.axvline(0, color='k', ls=':', alpha=0.5)
            ax.set_xlim(min(bars[metric].min() - 0.1, -0.2), 1.0)

    plt.tight_layout()
    path = f'{PLOT_DIR}/{plot_num}_external_error_bars.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_residual_distribution(df_ext, plot_num=70):
    """Residual distribution: training LOO vs external."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (model_name, res_col) in zip(axes, [
            ('SISSO Full', 'residual_SISSO'), ('SISSO Robust', 'residual_SISSO_robust'),
            ('M3', 'residual_M3')]):
        # By source
        for src in df_ext['source'].unique():
            mask = df_ext['source'] == src
            resids = df_ext.loc[mask, res_col].values
            ax.hist(resids, bins=15, alpha=0.5, label=f'{src} (n={mask.sum()})',
                    color=SOURCE_COLORS.get(src, '#333333'))

        ax.axvline(0, color='k', ls='--', alpha=0.5)
        bias = df_ext[res_col].mean()
        ax.axvline(bias, color='red', ls=':', alpha=0.8,
                   label=f'Mean bias: {bias:.1f} MPa')

        ax.set_xlabel('Residual (Exp − Pred) (MPa)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{model_name} — Residuals', fontsize=13)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = f'{PLOT_DIR}/{plot_num}_external_residuals.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# SECTION 8: MAIN EXECUTION
# ============================================================

def main():
    t0 = time.time()

    # 1. Fit models on training data
    models = fit_training_models()

    # 2. Load external data
    df_ext = load_all_external_data(models['df_train'])

    if df_ext.empty:
        print("\nERROR: No external data loaded. Exiting.")
        return

    # 3. Generate predictions
    df_ext = predict_all_external(df_ext, models)

    # 4. Evaluate (returns per-source results + tier-separated results)
    error_results, tier_results = evaluate_models(df_ext)

    # 5. HP slope analysis
    df_slopes = hp_slope_analysis(df_ext)

    # 5b. C_eff sensitivity analysis on Huang HV data
    df_huang_only = df_ext[df_ext['source'] == 'Huang2019'].copy()
    if not df_huang_only.empty and 'HV_exp' in df_huang_only.columns:
        df_ceff_sens = ceff_sensitivity_analysis(df_huang_only, models)
    else:
        df_ceff_sens = pd.DataFrame()

    # 6. Visualize
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_parity(df_ext, models, plot_num=67)
    plot_hp_slopes(df_ext, df_slopes, plot_num=68)
    plot_error_bars(error_results, plot_num=69)
    plot_residual_distribution(df_ext, plot_num=70)

    # 7. Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    out_csv = f'{RESULTS_DIR}/external_validation_results.csv'
    df_ext.to_csv(out_csv, index=False)
    print(f"  Saved predictions: {out_csv}")

    if not df_slopes.empty:
        slopes_csv = f'{RESULTS_DIR}/external_hp_slope_comparison.csv'
        df_slopes.to_csv(slopes_csv, index=False)
        print(f"  Saved HP slopes: {slopes_csv}")

    if not df_ceff_sens.empty:
        sens_csv = f'{RESULTS_DIR}/external_ceff_sensitivity.csv'
        df_ceff_sens.to_csv(sens_csv, index=False)
        print(f"  Saved C_eff sensitivity: {sens_csv}")

    if tier_results:
        tier_csv = f'{RESULTS_DIR}/external_tier_results.csv'
        pd.DataFrame(tier_results).to_csv(tier_csv, index=False)
        print(f"  Saved tier results: {tier_csv}")

    # 8. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  External dataset: {len(df_ext)} data points from {df_ext['source'].nunique()} sources")
    print(f"  Unique alloys: {df_ext['alloy'].nunique()}")
    print(f"  Grain size range: {df_ext['GrainSize'].min():.1f} – {df_ext['GrainSize'].max():.1f} μm")
    print(f"  YS range: {df_ext['YS_exp'].min():.0f} – {df_ext['YS_exp'].max():.0f} MPa")

    print(f"\n  --- Overall Performance ---")
    for model_name, pred_col in [('SISSO Full', 'YS_SISSO'),
                                  ('SISSO Robust', 'YS_SISSO_robust'),
                                  ('M3', 'YS_M3')]:
        r2 = r2_score(df_ext['YS_exp'], df_ext[pred_col])
        rmse = np.sqrt(mean_squared_error(df_ext['YS_exp'], df_ext[pred_col]))
        mae = mean_absolute_error(df_ext['YS_exp'], df_ext[pred_col])
        bias = np.mean(df_ext[pred_col] - df_ext['YS_exp'])
        print(f"    {model_name:>14s}: R²={r2:.3f}, RMSE={rmse:.1f}, MAE={mae:.1f}, bias={bias:+.1f} MPa")

    # Compare to training LOO
    print(f"\n  --- Training LOO (for reference) ---")
    print(f"    SISSO Full:   LOO R² = 0.671, RMSE = 46.5 MPa")
    print(f"    SISSO Robust: LOO R² = 0.626, RMSE = 49.6 MPa")
    print(f"    M3:           LOO R² = 0.652, RMSE = 47.8 MPa")

    # Data quality summary
    print(f"\n  --- Data Quality Notes ---")
    print(f"    Verified (open-access papers): Schneider2021 (CrFeNi), Huang2019 (HV)")
    print(f"    HP-derived (published params): Otto2013 (CoCrFeMnNi)")
    print(f"    Aggregated (heterogeneous):    Citrine MPEA dataset")
    print(f"    CAVEAT: Schneider data is COMPRESSION; Otto is TENSION")
    print(f"    CAVEAT: HV -> YS conversion uses C_eff = {C_EFF} (our estimate)")

    if not df_slopes.empty:
        print(f"\n  --- HP Slopes (k_HP, MPa·μm^1/2) ---")
        print(f"    {'Alloy':<16s} {'Exp':>8s} {'SISSO':>8s} {'Robust':>8s} {'M3':>8s}")
        print("    " + "-" * 50)
        for _, row in df_slopes.iterrows():
            print(f"    {row['alloy']:<16s} {row['k_HP_exp']:8.0f} {row['k_HP_SISSO']:8.0f} "
                  f"{row['k_HP_SISSO_robust']:8.0f} {row['k_HP_M3']:8.0f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
