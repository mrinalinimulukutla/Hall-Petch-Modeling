# Physics_informed_ML_Hall_Petch_2 — staged reproduction
# `make help` lists every target. Stage order matches docs/reproducing.md.

PY    := python
S     := scripts
LATEX := cd paper && pdflatex -interaction=nonstopmode

.PHONY: help install test clean all verify \
        data diagnostics family1 bayesian family2 pca family3 sdgrain export-models \
        family4 fair family5 sisso pysr validation external audit grouped \
        hardness figures paper notebook report

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## pip install -r requirements.txt
	pip install -r requirements.txt

test:  ## Regression tests locking the manuscript's canonical values (~1 s)
	pytest tests/ -q

clean:  ## Remove LaTeX aux files and Python caches
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.spl paper/*.toc
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

# ---------------- stage 0: data ----------------

data:  ## Descriptors, VLC quantities, and the shared feature ladder
	$(PY) $(S)/00_data_preparation/eda_analysis.py
	$(PY) $(S)/00_data_preparation/vlc_corrected.py
	$(PY) $(S)/00_data_preparation/build_armote_inputs.py

diagnostics:  ## Pre-modelling diagnostics and within-replicate slopes
	$(PY) $(S)/00_data_preparation/eda_diagnostics.py
	$(PY) $(S)/01_family1_grain_size/eda_within_replicate_kHP.py

# ---------------- family 1: classical Hall-Petch ----------------

family1:  ## Nine grain-size scaling laws, YS and HV
	$(PY) $(S)/01_family1_grain_size/grain_size_scaling_analysis.py

bayesian:  ## Bayesian PSIS-LOO comparison of the scaling laws (needs PyMC)
	$(PY) $(S)/01_family1_grain_size/bayesian_scaling_analysis.py

# ---------------- family 2: physics descriptors ----------------

family2:  ## VLC / Labusch / Toda-Caraballo benchmark and redundancy audit
	$(PY) $(S)/02_family2_physics_descriptors/vlc_sss_analysis.py

pca:  ## Fold-contained PCA-OLS on the curated descriptor set
	$(PY) $(S)/02_family2_physics_descriptors/pca_ols_analysis.py

# ---------------- family 3: composition / processing ----------------

family3:  ## M-model hierarchy
	$(PY) $(S)/03_family3_composition_processing/composition_hp_analysis.py
	$(PY) $(S)/03_family3_composition_processing/kHP_composition_analysis.py

sdgrain:  ## M3 vs additive SD_grain vs the M15 interaction (headline YS result)
	$(PY) $(S)/03_family3_composition_processing/sdgrain_models.py

export-models:  ## Fitted-model pickles and coefficient tables
	$(PY) $(S)/03_family3_composition_processing/export_fitted_models.py

# ---------------- family 4: non-linear ML ----------------

family4:  ## Tuned panel and SHAP (slow)
	$(PY) $(S)/04_family4_nonlinear_ml/exhaustive_model_search.py
	$(PY) $(S)/04_family4_nonlinear_ml/xgboost_shap_analysis.py

fair:  ## Matched-input comparison at fixed settings, zero tuning
	$(PY) $(S)/04_family4_nonlinear_ml/fair_comparison.py

# ---------------- family 5: symbolic regression ----------------

sisso:  ## SISSO Full / Robust / v2 / +SD_grain
	$(PY) $(S)/05_family5_symbolic_regression/sisso_analysis.py
	$(PY) $(S)/05_family5_symbolic_regression/sisso_robust.py

pysr:  ## PySR feature x operator grid (needs Julia)
	$(PY) $(S)/05_family5_symbolic_regression/pysr_grid_analysis.py

family5: sisso pysr  ## All symbolic regression

# ---------------- validation protocol ----------------

grouped:  ## Pooled 5-fold / LOO / LOBO table for every headline model
	$(PY) $(S)/06_validation/grouped_validation.py

verified:  ## Recompute the manuscript quantities verifiable from this repo
	$(PY) $(S)/06_validation/verified_analysis.py

external:  ## Tiered literature stress test
	$(PY) $(S)/06_validation/external_validation.py

audit:  ## Singularity audit of every reported closed form
	$(PY) $(S)/06_validation/singularity_audit.py

validation: grouped verified external audit  ## Whole validation protocol

verify:  ## Fail fast if the manuscript drifts from its verified artifacts
	$(PY) $(S)/06_validation/validate_manuscript.py

# ---------------- hardness ----------------

hardness:  ## Tabor ratio, HV scaling, and HV-YS rank analysis
	$(PY) $(S)/07_hardness_tabor/hardness_analysis.py

# ---------------- outputs ----------------

figures:  ## Regenerate every publication figure from cached results (fast)
	$(PY) $(S)/figures/make_framework_overview.py
	$(PY) $(S)/figures/make_batch_colored_figures.py
	$(PY) $(S)/figures/_make_hv_ys_rank_fig.py
	$(PY) $(S)/figures/make_property_figures.py
	$(PY) $(S)/figures/make_provenance_figure.py
	$(PY) $(S)/figures/replot_fair_comparison_heatmap.py
	$(PY) $(S)/figures/make_restyled_figures.py

paper:  ## Build paper/main.pdf and paper/supplementary.pdf
	$(LATEX) main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
	$(LATEX) supplementary.tex && bibtex supplementary && pdflatex -interaction=nonstopmode supplementary.tex && pdflatex -interaction=nonstopmode supplementary.tex
	$(MAKE) clean

notebook:  ## Regenerate the analysis notebook from its generator
	$(PY) notebook/_generate_notebook.py

report:  ## Rebuild the comprehensive Word report from live results
	$(PY) report/generate_report.py

all: figures report notebook paper  ## Build every artifact from current results
