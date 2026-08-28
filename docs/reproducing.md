# Reproducing the analysis

Everything derives from one input: `data/raw/Grain_Size_Summary_v3.xlsx`
(94 rows, 19 columns). Each stage writes CSVs to `results/` and figures to
`analysis_plots/` and `paper/figures/`.

```bash
pip install -r requirements.txt
make test          # fastest useful check: ~1 s, recomputes every headline value
```

## Stage order

| # | Stage | Command | Time | Key outputs |
|---|-------|---------|------|-------------|
| 0 | Descriptors and feature ladder | `make data` | ~1 min | `data/derived/*.csv` |
| 0b | Pre-modelling diagnostics | `make diagnostics` | ~1 min | hull overlap, within-replicate slopes |
| 1 | Grain-size scaling laws | `make family1` | ~5 min | scaling-law comparison |
| 1b | Bayesian comparison *(archival)* | `make bayesian` | ~10 min | needs PyMC; see below |
| 2 | SSS benchmark and redundancy | `make family2 pca` | ~3 min | SSS table, fold-contained PCA-OLS |
| 3 | M-model hierarchy | `make family3 sdgrain export-models` | ~5 min | `sdgrain_m15_validation.csv`, coefficients |
| 4 | Non-linear ML | `make fair` | ~5 min | matched-input comparison |
| 4b | Tuned panel *(slow)* | `make family4` | ~35 min | legacy leaderboard, SHAP |
| 5 | Symbolic regression | `make family5` | ~25 min | SISSO and PySR results (PySR needs Julia) |
| 6 | Validation protocol | `make validation` | ~3 min | grouped CV, literature test, singularity audit |
| 7 | Hardness / Tabor | `make hardness` | ~2 min | C_eff, HV scaling, rank analysis |
| 8 | Figures | `make figures` | ~1 min | `paper/figures/`, `analysis_plots/` |
| 9 | Documents | `make all` | ~5 min | report, notebook, PDFs |
| — | Drift check | `make verify` | ~1 s | fails if the manuscript and artifacts disagree |

A failing test after a re-run means the environment changed, not the code.
Investigate before editing anything downstream.

## What is verified, and what is archival

The manuscript separates results that can be regenerated here from results
that cannot.

**Regenerated on demand.** The M-model hierarchy including M15, the Family 1
baselines for both targets, fold-contained PCA-OLS, the matched-input Family 4
comparison, the tiered literature stress test, the Tabor analysis, and the
dataset audit. `make test` recomputes these from `data/derived` and asserts
them against the values printed in the paper.

**Archival.** Three analyses are reported as archival rather than
confirmatory, because the artifacts needed to reproduce them were not
retained:

- **Bayesian PSIS-LOO stacking weights.** The PyMC posterior draws are gone.
  The frequentist information criteria for the same models are recomputed.
- **Nested ARMOTE-CV panel.** Per-fold models and Optuna studies are archived
  separately (~1.5 GB). The manuscript uses it as secondary evidence only.
- **Outer-loop PySR performance.** Equation structures were selected after
  viewing complete-data search fronts, so the reported scores are
  post-selection fixed-form CV, not unbiased symbolic-discovery estimates.
  The manuscript labels them accordingly and does not rank them against the
  pre-specified models.

## Determinism

Seeds are fixed inside each script (5-fold uses seed 42). SISSO reruns are
deterministic. PySR is evolutionary and can return different, equally scoring
expressions between runs; what the tests lock is the cross-validated
performance of the *reported* structures, not the expression strings.

## Environment

`requirements.txt` pins the analysis stack. Two optional extras:

- `pymc` and `arviz` for stage 1b
- Julia and `pysr` for the PySR grid

Neither is needed for `make test`, `make figures`, `make report` or
`make paper`.
