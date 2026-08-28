#!/usr/bin/env python3
"""
Within-replicate Hall-Petch slope analysis.

For each group of alloys at IDENTICAL composition but different grain size,
fit a local k_HP. Then test heterogeneity across groups via Cochran's Q.

The paper's claim is that k_HP is composition-independent (R^2 = 0.006 from
regressing per-alloy k_HP against composition). If so, the local slopes from
the 9 composition-replicate groups should agree to within their measurement
noise. If they don't, k_HP varies with composition and the paper underclaims.

Outputs:
  - analysis_plots/74_within_replicate_kHP.png
  - within_replicate_kHP_summary.txt
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for _config)
from _config import REPO_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, PLOTS_DIR, PAPER_DIR, PAPER_FIG_DIR
BASE = str(REPO_ROOT)
PLOT_DIR = f"{PLOTS_DIR}"
DATA_CSV = f"{DATA_DIR}/data_with_descriptors.csv"

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "V"]
RMS_SD_DEFAULT = 15.3  # rms of SD_YS across the 80 alloys with reported uncertainty

df = pd.read_csv(DATA_CSV).dropna(subset=["YS"]).reset_index(drop=True)
n_total = len(df)
median_sd = df["SD_YS"].median()

summary_lines = []
def log(msg=""):
    print(msg)
    summary_lines.append(msg)

log("=" * 72)
log("WITHIN-REPLICATE HALL-PETCH SLOPE ANALYSIS")
log("=" * 72)
log(f"n_total = {n_total} alloys (YS available)")
log(f"Median SD_YS = {median_sd:.1f} MPa, rms SD_YS = {RMS_SD_DEFAULT:.1f} MPa")

# Global Hall-Petch fit (baseline reference)
x_all = df["d_inv_sqrt"].values
y_all = df["YS"].values
slope_global, intercept_global, r_global, p_global, se_global = stats.linregress(x_all, y_all)
log(f"\nGlobal HP fit (n={n_total}):")
log(f"  σ_0   = {intercept_global:.1f} MPa")
log(f"  k_HP  = {slope_global:.0f} MPa·μm^(1/2),  SE = {se_global:.0f}")
log(f"  R²    = {r_global**2:.3f},  p = {p_global:.2e}")

# Identify replicate groups
df = df.assign(_compkey=df[ELEMENTS].apply(lambda r: tuple(r.values), axis=1))
groups = df.groupby("_compkey")

results = []
gid = 0
for compkey, g in groups:
    if len(g) < 2:
        continue
    gid += 1
    g = g.sort_values("d_inv_sqrt")
    x = g["d_inv_sqrt"].values
    y = g["YS"].values
    sd_y = g["SD_YS"].fillna(median_sd).fillna(RMS_SD_DEFAULT).values

    # OLS slope (exact for n=2)
    slope, intercept, *_ = stats.linregress(x, y)

    # SE(k) via per-alloy noise model: Var(k) = mean(SD_YS^2) / Sxx
    sxx = np.sum((x - x.mean()) ** 2)
    sigma2_y = np.mean(sd_y ** 2)
    se_k = np.sqrt(sigma2_y / sxx) if sxx > 0 else np.nan

    d_range = g["GrainSize"].max() - g["GrainSize"].min()
    comp_str = "/".join(f"{int(c):d}" for c in compkey)

    results.append({
        "gid": gid,
        "comp_str": comp_str,
        "compkey": compkey,
        "alloys": list(g["Alloy"].values),
        "batches": list(g["Iteration"].values),
        "n": len(g),
        "d_min": g["GrainSize"].min(),
        "d_max": g["GrainSize"].max(),
        "d_range": d_range,
        "x": x, "y": y, "sd_y": sd_y,
        "k_local": slope,
        "se_k": se_k,
        "sigma0": intercept,
    })

# Print per-group fits
log("\n" + "-" * 72)
log("PER-GROUP HP FITS")
log("-" * 72)
log(f"{'G':>3s} {'Composition (at%)':<28s} {'n':>3s} {'d range (μm)':>14s} "
    f"{'k_local':>9s} {'SE(k)':>8s} {'σ_0':>7s}  batches")
for r in results:
    log(f"G{r['gid']:<2d} {r['comp_str']:<28s} {r['n']:>3d} "
        f"{r['d_min']:>5.0f}-{r['d_max']:<7.0f} "
        f"{r['k_local']:>+9.0f} {r['se_k']:>8.0f} {r['sigma0']:>7.0f}  "
        f"{','.join(set(r['batches']))}")

# Heterogeneity test (Cochran's Q)
log("\n" + "-" * 72)
log("HETEROGENEITY TEST (Cochran's Q on per-group k_HP)")
log("-" * 72)
valid = [r for r in results if r["d_range"] > 2 and np.isfinite(r["se_k"]) and r["se_k"] > 0]
log(f"\nValid groups (d_range > 2 μm, finite SE): {len(valid)} of {len(results)}")

if len(valid) >= 2:
    ks = np.array([r["k_local"] for r in valid])
    ses = np.array([r["se_k"] for r in valid])
    weights = 1.0 / ses ** 2
    k_pooled = np.sum(weights * ks) / np.sum(weights)
    se_pooled = 1.0 / np.sqrt(np.sum(weights))
    Q = np.sum(weights * (ks - k_pooled) ** 2)
    dof = len(valid) - 1
    pQ = 1.0 - stats.chi2.cdf(Q, dof)
    I2 = max(0.0, (Q - dof) / Q) * 100 if Q > 0 else 0.0

    log(f"\nFixed-effects pooled k_HP = {k_pooled:.0f} ± {se_pooled:.0f} MPa·μm^(1/2)")
    log(f"  (compare to global OLS k_HP = {slope_global:.0f})")
    log(f"\nCochran's Q = {Q:.2f}  on dof = {dof},  p = {pQ:.3f}")
    log(f"I² = {I2:.1f}%   (<25% low, 25-50% moderate, 50-75% substantial, >75% high)")
    if pQ < 0.05:
        log(f"  ⇒ SIGNIFICANT heterogeneity: k_HP varies more across compositions")
        log(f"    than measurement noise predicts. Paper's k_HP-is-constant claim weakens.")
    else:
        log(f"  ⇒ NO significant heterogeneity at α=0.05.")
        log(f"    k_HP variation across replicate groups is consistent with noise;")
        log(f"    paper's claim that k_HP is composition-independent is supported.")

    log(f"\nObserved range of k_local:  {ks.min():.0f} to {ks.max():.0f} MPa·μm^(1/2)")
    log(f"Mean ± SD of k_local:       {ks.mean():.0f} ± {ks.std(ddof=1):.0f}")
    log(f"Coefficient of variation:   {ks.std(ddof=1)/abs(ks.mean())*100:.1f}%")
else:
    log("\nNot enough valid groups for heterogeneity test.")
    Q, pQ, I2, k_pooled, se_pooled = np.nan, np.nan, np.nan, np.nan, np.nan

# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

# (A) Forest plot
ax = fig.add_subplot(gs[0, 0])
sorted_res = sorted(results, key=lambda r: r["k_local"])
for i, r in enumerate(sorted_res):
    is_valid = r["d_range"] > 2 and np.isfinite(r["se_k"]) and r["se_k"] > 0
    color = "steelblue" if is_valid else "gray"
    ax.errorbar(r["k_local"], i,
                xerr=1.96 * r["se_k"] if is_valid else 0,
                fmt="o", color=color, markersize=8, capsize=3)
    ax.text(r["k_local"], i + 0.15, f"  n={r['n']}, Δd={r['d_range']:.0f}μm",
            fontsize=8, va="bottom")
ax.axvline(slope_global, color="red", ls="--", lw=1.5,
           label=f"Global k_HP = {slope_global:.0f}")
if np.isfinite(k_pooled):
    ax.axvline(k_pooled, color="green", ls=":", lw=1.5,
               label=f"Pooled k = {k_pooled:.0f}")
    ax.axvspan(k_pooled - 1.96 * se_pooled, k_pooled + 1.96 * se_pooled,
               color="green", alpha=0.1)
ax.set_yticks(range(len(sorted_res)))
ax.set_yticklabels([f"G{r['gid']}" for r in sorted_res])
ax.set_xlabel("Local k_HP (MPa·μm^(1/2))", fontsize=11)
title = f"Per-group HP slopes (forest plot)"
if np.isfinite(Q):
    title += f"\nQ={Q:.2f}, p={pQ:.3f}, I²={I2:.0f}%"
ax.set_title(title, fontsize=11)
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.3, axis="x")

# (B) HP plot with replicate-group segments
ax = fig.add_subplot(gs[0, 1:])
ax.scatter(x_all, y_all, c="lightgray", s=20, alpha=0.5, label="all alloys")
cmap = plt.cm.tab10
for i, r in enumerate(results):
    color = cmap(i % 10)
    ax.scatter(r["x"], r["y"], c=[color], s=80, alpha=0.9,
               edgecolors="k", linewidths=0.5,
               label=f"G{r['gid']}: {r['comp_str']}")
    xs = np.linspace(r["x"].min(), r["x"].max(), 20)
    ax.plot(xs, r["sigma0"] + r["k_local"] * xs, "-", color=color, lw=1.8, alpha=0.8)
xfit = np.linspace(x_all.min(), x_all.max(), 100)
ax.plot(xfit, intercept_global + slope_global * xfit, "r--", lw=2,
        label=f"Global: σ_0={intercept_global:.0f} + {slope_global:.0f}·d^(-1/2)")
ax.set_xlabel("d^(-1/2) (μm^(-1/2))", fontsize=11)
ax.set_ylabel("YS (MPa)", fontsize=11)
ax.set_title("Hall-Petch: replicate-group segments vs global fit", fontsize=12)
ax.legend(fontsize=7, loc="best", ncol=2)
ax.grid(True, alpha=0.3)

# (C-E) k_local vs candidate composition descriptors
def k_vs_descriptor(ax, descr_vals, label, results):
    ks = [r["k_local"] for r in results]
    ax.scatter(descr_vals, ks, s=80, alpha=0.7, edgecolors="k")
    for v, k, r in zip(descr_vals, ks, results):
        ax.annotate(f"G{r['gid']}", (v, k), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(slope_global, color="red", ls="--", label=f"global k = {slope_global:.0f}")
    if len(set(descr_vals)) > 1:
        rho, p_rho = stats.spearmanr(descr_vals, ks)
        ax.set_title(f"k_local vs {label}\n(Spearman ρ={rho:+.2f}, p={p_rho:.2f})",
                     fontsize=11)
    else:
        ax.set_title(f"k_local vs {label}", fontsize=11)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Local k_HP", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
v_vals = [r["compkey"][ELEMENTS.index("V")] for r in results]
k_vs_descriptor(ax, v_vals, "V content (at%)", results)

ax = fig.add_subplot(gs[1, 1])
cr_vals = [r["compkey"][ELEMENTS.index("Cr")] for r in results]
k_vs_descriptor(ax, cr_vals, "Cr content (at%)", results)

ax = fig.add_subplot(gs[1, 2])
ncomp_vals = [sum(1 for c in r["compkey"] if c > 0) for r in results]
k_vs_descriptor(ax, ncomp_vals, "n_comp (nonzero elements)", results)

plt.suptitle(
    "Within-replicate Hall-Petch slope analysis: "
    "is k_HP composition-dependent?",
    fontsize=13, y=1.00
)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/74_within_replicate_kHP.png", dpi=150)
plt.close()
log("\nSaved: analysis_plots/74_within_replicate_kHP.png")

with open(f"{RESULTS_DIR}/within_replicate_kHP_summary.txt", "w") as f:
    f.write("\n".join(summary_lines) + "\n")
log("Saved: within_replicate_kHP_summary.txt")
