#!/usr/bin/env python3
"""Generate two candidate figures for Section 2.2 (Dataset composition
and property statistics).

Option 3: Pairwise property scatter (YS vs HV, YS vs d^(-1/2), HV vs d^(-1/2))
Option 4: Property histograms (YS, HV, d) stacked/overlaid by batch
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
import _figstyle as S
S.apply()
BASE = str(REPO_ROOT)
BATCHES = ["BBA", "BBB", "BBC", "CBA", "CBB", "CBC"]
BATCH_COLORS = {b: c for b, (c, _) in S.BATCH.items()}
MARKERS = {b: m for b, (_, m) in S.BATCH.items()}

df = pd.read_csv(f"{DATA_DIR}/data_with_descriptors.csv").dropna(subset=["YS"]).reset_index(drop=True)
print(f"Loaded {len(df)} alloys")

# ============================================================
# Option 3: pairwise property scatter
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(S.W_FULL, 2.45))

# (a) YS vs HV
ax = axes[0]
mask_hv = df["HV"].notna()
for batch in BATCHES:
    m = (df["Iteration"].values == batch) & mask_hv.values
    if not m.any():
        continue
    ax.scatter(df.loc[m, "HV"], df.loc[m, "YS"],
               c=BATCH_COLORS[batch], marker=MARKERS[batch],
               s=55, alpha=0.8, edgecolors="k", linewidth=0.4, label=batch)
r_hv = df[mask_hv][["HV", "YS"]].corr().iloc[0, 1]
ax.set_xlabel("Vickers hardness HV")
ax.set_ylabel("Yield strength YS (MPa)")
ax.set_title(f"(a) YS vs HV   (Pearson r = {r_hv:.2f})", loc="left")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, loc="lower right", framealpha=0.9)

# (b) YS vs d^(-1/2)  — Hall-Petch
ax = axes[1]
for batch in BATCHES:
    m = df["Iteration"].values == batch
    if not m.any():
        continue
    ax.scatter(df.loc[m, "d_inv_sqrt"], df.loc[m, "YS"],
               c=BATCH_COLORS[batch], marker=MARKERS[batch],
               s=55, alpha=0.8, edgecolors="k", linewidth=0.4, label=batch)
slope, intercept, r_ys, p_ys, _ = stats.linregress(df["d_inv_sqrt"], df["YS"])
xfit = np.linspace(df["d_inv_sqrt"].min(), df["d_inv_sqrt"].max(), 50)
ax.plot(xfit, intercept + slope * xfit, "k--", lw=1.5,
        label=f"HP fit: $R^2$ = {r_ys**2:.2f}")
ax.set_xlabel(r"$d^{-1/2}$ ($\mu m^{-1/2}$)")
ax.set_ylabel("Yield strength YS (MPa)")
ax.set_title(f"(b) Hall-Petch (YS)", loc="left")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", framealpha=0.9)

# (c) HV vs d^(-1/2)
ax = axes[2]
for batch in BATCHES:
    m = (df["Iteration"].values == batch) & mask_hv.values
    if not m.any():
        continue
    ax.scatter(df.loc[m, "d_inv_sqrt"], df.loc[m, "HV"],
               c=BATCH_COLORS[batch], marker=MARKERS[batch],
               s=55, alpha=0.8, edgecolors="k", linewidth=0.4, label=batch)
slope_h, intercept_h, r_hv2, _, _ = stats.linregress(
    df.loc[mask_hv, "d_inv_sqrt"], df.loc[mask_hv, "HV"])
ax.plot(xfit, intercept_h + slope_h * xfit, "k--", lw=1.5,
        label=f"HP fit: $R^2$ = {r_hv2**2:.2f}")
ax.set_xlabel(r"$d^{-1/2}$ ($\mu m^{-1/2}$)")
ax.set_ylabel("Vickers hardness HV")
ax.set_title(f"(c) Hall-Petch (HV)", loc="left")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", framealpha=0.9)
plt.savefig(f"{PAPER_FIG_DIR}/fig_property_pairwise.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig_property_pairwise.png")

# ============================================================
# Option 4: property histograms stacked by batch (2x2, 1:1 overall)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Per-batch arrays for stacked histograms
def stacked_hist(ax, col, bins, xlabel, log_x=False):
    data_by_batch = []
    colors = []
    labels = []
    for batch in BATCHES:
        m = (df["Iteration"].values == batch) & df[col].notna().values
        vals = df.loc[m, col].values
        if len(vals) == 0:
            continue
        data_by_batch.append(vals)
        colors.append(BATCH_COLORS[batch])
        labels.append(f"{batch} ($n$={len(vals)})")
    ax.hist(data_by_batch, bins=bins, stacked=True, color=colors,
            edgecolor="k", linewidth=0.4, label=labels, alpha=0.95)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of alloys", fontsize=12)
    if log_x:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="y", which="both")
    else:
        ax.grid(True, alpha=0.3, axis="y")
    ax.legend(ncol=2, framealpha=0.9)

# (a) YS histogram
ax = axes[0, 0]
stacked_hist(ax, "YS", bins=np.arange(140, 580, 30),
             xlabel="Yield strength YS (MPa)")
mean_ys, std_ys = df["YS"].mean(), df["YS"].std()
ax.set_title(f"(a) YS distribution   ({mean_ys:.0f} $\\pm$ {std_ys:.0f} MPa)",
             fontsize=12, loc="left")

# (b) HV histogram
ax = axes[0, 1]
stacked_hist(ax, "HV", bins=np.arange(50, 240, 12),
             xlabel="Vickers hardness HV")
mean_hv, std_hv = df["HV"].mean(), df["HV"].std()
ax.set_title(f"(b) HV distribution   ({mean_hv:.0f} $\\pm$ {std_hv:.0f})",
             fontsize=12, loc="left")

# (c) grain size histogram on log scale
ax = axes[1, 0]
log_bins = np.logspace(np.log10(10), np.log10(250), 18)
stacked_hist(ax, "GrainSize", bins=log_bins,
             xlabel=r"Grain size $d$ ($\mu m$)", log_x=True)
mean_d, std_d = df["GrainSize"].mean(), df["GrainSize"].std()
ax.set_title(f"(c) Grain-size distribution (log scale)   ({mean_d:.0f} $\\pm$ {std_d:.0f} μm)",
             fontsize=12, loc="left")

# (d) Tabor ratio HV(MPa)/YS distribution
ax = axes[1, 1]
mask_both = df["HV"].notna() & df["YS"].notna()
df2 = df[mask_both].copy()
df2["Ceff"] = df2["HV"] * 9.807 / df2["YS"]
ceff_bins = np.linspace(2, 10, 25)
data_by_batch = []
colors = []
labels = []
for batch in BATCHES:
    m = df2["Iteration"].values == batch
    vals = df2.loc[m, "Ceff"].values
    if len(vals) == 0:
        continue
    data_by_batch.append(vals)
    colors.append(BATCH_COLORS[batch])
    labels.append(f"{batch} ($n$={len(vals)})")
ax.hist(data_by_batch, bins=ceff_bins, stacked=True, color=colors,
        edgecolor="k", linewidth=0.4, label=labels, alpha=0.95)
ax.axvline(3.0, color="k", ls="--", lw=1.2, label="classical Tabor $C=3$")
mean_c, std_c = df2["Ceff"].mean(), df2["Ceff"].std()
ax.set_xlabel(r"Effective Tabor factor $C_\mathrm{eff} = \mathrm{HV(MPa)}/\sigma_y$", fontsize=12)
ax.set_ylabel("Number of alloys", fontsize=12)
ax.set_title(f"(d) Tabor ratio distribution   ({mean_c:.2f} $\\pm$ {std_c:.2f})",
             fontsize=12, loc="left")
ax.grid(True, alpha=0.3, axis="y")
ax.legend(ncol=2, framealpha=0.9)
plt.savefig(f"{PAPER_FIG_DIR}/fig_property_histograms.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig_property_histograms.png")
