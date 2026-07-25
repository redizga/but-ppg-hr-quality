.PHONY: help setup test lint prepare splits baselines cnn opentslm sigma lowdata galaxy-prep galaxy-eval evaluate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:      ## install the reproducibility core (editable)
	pip install -e ".[dev]"

test:       ## run E0 infrastructure tests
	pytest -q

lint:       ## ruff check
	ruff check src scripts tests

prepare:    ## E1: download + preprocess BUT PPG, build registry
	python scripts/prepare_but_ppg.py

splits:     ## E1: subject-wise 60/20/20 split + overlap check
	python scripts/make_splits.py --registry data/processed/registry.csv --out splits/but_ppg_60_20_20.json --seed 42

baselines:  ## E3: feature-based LogReg/XGBoost
	python scripts/train_baseline.py --config models/baseline_features

cnn:        ## E4: 1D-CNN / ResNet1D
	python scripts/train_cnn.py --config models/cnn1d

opentslm:   ## E5: OpenTSLM (no-ECG and ECG-init)
	python scripts/train_opentslm.py --config models/opentslm model.ecg_init=false
	python scripts/train_opentslm.py --config models/opentslm model.ecg_init=true

sigma:      ## E6: SIGMA-PPG fine-tune (PPG-only)
	python scripts/train_sigma.py --config models/sigma_ppg

lowdata:    ## E8: low-data experiment
	python scripts/run_low_data.py --config experiment/low_data

galaxy-prep: ## E7: prepare GalaxyPPG
	python scripts/prepare_galaxy_ppg.py --config data/galaxy_ppg

galaxy-eval: ## E7: external validation on GalaxyPPG
	python scripts/eval_external_galaxy.py --config data/galaxy_ppg

evaluate:   ## E2: score all predictions on the fixed test set
	python scripts/evaluate.py --all
