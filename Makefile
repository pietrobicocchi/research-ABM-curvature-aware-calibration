PYTHON ?= python
export PYTHONPATH := src:.

EXPERIMENTS = \
	mfsir_calibration_path \
	mfsir_laplace_vs_posterior \
	mfsir_ggn_vs_fd_hessian \
	mfsir_spectrum_loadings \
	mfsir_counterfactual \
	mfsir_sloppy_direction_check \
	netsir_observation_window \
	netsir_estimator_comparison \
	netsir_gradient_robustness

ZIP_EXCLUDE = "*.git*" "*/.venv/*" ".venv/*" "*/__pycache__/*" "*.pyc" "*/.pytest_cache/*" "*.DS_Store" "outputs/*" "supplementary_code.zip"

.PHONY: figures compute all check test zip clean

# Plot every figure from the saved results in results/.
figures:
	@for e in $(EXPERIMENTS); do $(PYTHON) -m experiments.$$e plot || exit 1; done

# Recompute every saved result (overwrites results/).
compute:
	@for e in $(EXPERIMENTS); do echo "== $$e"; $(PYTHON) -m experiments.$$e compute || exit 1; done

all: compute figures

check:
	$(PYTHON) scripts/check_claims.py

test:
	$(PYTHON) -m pytest -q tests

zip:
	rm -f supplementary_code.zip
	zip -qr supplementary_code.zip . -x $(ZIP_EXCLUDE)
	@ls -lh supplementary_code.zip

clean:
	rm -rf figures supplementary_code.zip
