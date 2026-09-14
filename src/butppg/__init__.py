"""butppg — reproducible PPG signal-quality & heart-rate study on BUT PPG v2.0.0.

The package is intentionally thin at E0: it ships the reproducibility spine
(config loading, seeding, run manifests, logging, record registry) that every
later epic (data prep, baselines, OpenTSLM, SIGMA-PPG, external validation)
plugs into. Model/data internals arrive with their respective epics.
"""

__version__ = "0.1.0"
