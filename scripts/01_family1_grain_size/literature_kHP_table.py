#!/usr/bin/env python3
"""
Literature k_HP Comparison Table
=================================
Curated reference table of published k_HP values for FCC HEAs and pure
FCC metals, juxtaposed with our M3 and SR-derived equivalents.

Sources (with DOI in references.bib):
  - Otto et al. 2013, CoCrFeMnNi          k_HP = 494 MPa·μm^(1/2)
  - Yoshida et al. 2017, CoCrFeMnNi       k_HP = 538 MPa·μm^(1/2)
  - Sun et al. 2018, CoCrNi               k_HP = 265 MPa·μm^(1/2)
  - Schneider et al. 2020, CoCrFeMnNi     k_HP = 680 MPa·μm^(1/2)
  - LaRosa et al. 2019, Al0.3CoCrFeNi     k_HP = 824 MPa·μm^(1/2)
  - This work (M3)                        k_HP = 766 MPa·μm^(1/2)
  - This work (SISSO Full)                k_HP ≈ implied by 9356/dS_mix term
  - Pure FCC metals (Cu, Al, Ni avg)      k_HP ≈ 110–210 MPa·μm^(1/2)

Outputs
-------
  results/literature_kHP_table.csv
  analysis_plots/88_literature_kHP.png
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import RESULTS_DIR, PLOTS_DIR

LITERATURE = [
    # (system, k_HP, citation_key, note)
    ('CoCrFeMnNi (Otto 2013)',          494,  'Otto2013',    'tension, RT, fine-grained set'),
    ('CoCrFeMnNi (Yoshida 2017)',       538,  'Yoshida2017', 'wide GS range, 296 K'),
    ('CoCrFeMnNi (Schneider 2020)',     680,  'Schneider2020','wider GS sweep'),
    ('CoCrNi (Sun 2018)',               265,  'Sun2018',     'medium-entropy, tension'),
    ('Al0.3CoCrFeNi (LaRosa 2019)',     824,  'LaRosa2019',  'Al-bearing FCC HEA'),
    ('Pure Cu (avg)',                   110,  'Hall1951',    'baseline pure FCC'),
    ('Pure Al (avg)',                   150,  'Petch1953',   'baseline pure FCC'),
    ('Pure Ni (avg)',                   210,  'George2019',  'baseline pure FCC'),
    ('This work — M3',                  766,  'this_work',   '93 FCC HEAs, all 7 non-Ni elements'),
    ('This work — SISSO Full',          np.nan, 'this_work', 'implied 9356/dS_mix term; not directly comparable'),
    ('This work — SISSO Robust',        np.nan, 'this_work', 'implied 9837/dS_mix term'),
    ('This work — PySR compact YS',     np.nan, 'this_work', 'implied via 4.29*dH_mix*SD_grain/d^2 product'),
]

df = pd.DataFrame(LITERATURE, columns=['System', 'k_HP_MPa_um_half', 'Citation', 'Note'])
df.to_csv(f'{RESULTS_DIR}/literature_kHP_table.csv', index=False)
print(f"Wrote {RESULTS_DIR}/literature_kHP_table.csv")
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
plot_df = df.dropna(subset=['k_HP_MPa_um_half']).copy()
colors = ['#888' if 'Pure' in s else
          '#4c72b0' if 'this work' not in s.lower() else
          '#dd8452' for s in plot_df['System']]
ax.barh(plot_df['System'][::-1], plot_df['k_HP_MPa_um_half'][::-1],
        color=colors[::-1], edgecolor='black')
ax.axvline(766, color='#dd8452', linestyle='--',
           label='this work (M3) = 766')
ax.set_xlabel('k_HP  (MPa · μm^(1/2))')
ax.set_title('Hall–Petch coefficient: literature vs this work')
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/88_literature_kHP.png', dpi=150)
plt.close()
print(f"Wrote {PLOTS_DIR}/88_literature_kHP.png")
