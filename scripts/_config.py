"""Shared path configuration for all analysis scripts.

Every script under scripts/ imports REPO_ROOT, DATA_DIR, RESULTS_DIR, etc.
from this module, so the entire repo can be moved or cloned without editing
any individual script.

Scripts are organized by the five model families of the manuscript:
Family 1 classical Hall-Petch -> Family 2 physics descriptors ->
Family 3 composition/processing -> Family 4 non-linear ML ->
Family 5 symbolic regression, plus the validation protocol and the
hardness/Tabor synthesis. Importing this module also appends every scripts/ subdirectory
to sys.path, so cross-script imports (e.g. ``from external_validation
import ...``) keep working regardless of which tier folder a script lives in.

Convention:
    DATA_DIR / 'data_with_descriptors.csv'        # derived features
    RAW_DATA_DIR / 'Grain_Size_Summary_v3.xlsx'   # raw experimental data
    RESULTS_DIR / 'sisso_results.csv'             # output of analysis scripts
    PLOTS_DIR / '57_sisso_full_parity.png'        # analysis figures
    PAPER_FIG_DIR / 'fig08_symbolic_pareto.png'   # publication figures
"""
import sys
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = REPO_ROOT / 'data' / 'derived'
RAW_DATA_DIR   = REPO_ROOT / 'data' / 'raw'
RESULTS_DIR    = REPO_ROOT / 'results'
PLOTS_DIR      = REPO_ROOT / 'analysis_plots'
NOTEBOOK_DIR   = REPO_ROOT / 'notebook'
PAPER_DIR      = REPO_ROOT / 'paper'
PAPER_FIG_DIR  = REPO_ROOT / 'paper' / 'figures'
REPORT_DIR     = REPO_ROOT / 'report'
DOCS_DIR       = REPO_ROOT / 'docs'

SCRIPTS_DIR    = REPO_ROOT / 'scripts'

# Family folders, in pipeline order.
SCRIPT_SUBDIRS = [
    '00_data_preparation',
    '01_family1_grain_size',
    '02_family2_physics_descriptors',
    '03_family3_composition_processing',
    '04_family4_nonlinear_ml',
    '05_family5_symbolic_regression',
    '06_validation',
    '07_hardness_tabor',
    'figures',
]

# Make every family folder importable so scripts can import each other
# (e.g. sisso_robust.py imports descriptor tables from external_validation.py).
for _sub in SCRIPT_SUBDIRS:
    _p = SCRIPTS_DIR / _sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))


def find_script(name):
    """Return the Path of a script by filename, searching all tier folders.

    Accepts 'sisso_analysis.py' or 'sisso_analysis'. Raises FileNotFoundError
    if the script does not exist anywhere under scripts/.
    """
    if not name.endswith('.py'):
        name += '.py'
    direct = SCRIPTS_DIR / name
    if direct.exists():
        return direct
    for hit in sorted(SCRIPTS_DIR.rglob(name)):
        return hit
    raise FileNotFoundError(f'{name} not found under {SCRIPTS_DIR}')


# Ensure output dirs exist when a fresh clone is run
for d in (RESULTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Backward-compatibility string alias (some scripts use f'{BASE}/...')
BASE = str(REPO_ROOT)
