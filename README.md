# Kepler Exoplanet Classification

CMPS 261 (Machine Learning) course project, American University of Beirut — Track A.

**Team:** Joe Khoury, Andy Sleiman, Anthony Merhy

## Problem
NASA's Kepler telescope flagged ~9,500 "Kepler Objects of Interest" (KOIs): stars whose brightness dips in a way that could be a planet passing in front of them. Many are false positives (eclipsing binary stars, noise, light from a neighbouring star).
We classify each KOI as a real planet or a false positive using **only measurements available before human vetting**, so the model can help scientists rank which detections deserve follow-up.

## Approach
- **Labels:** CONFIRMED (1) vs FALSE POSITIVE (0); 7,587 objects. The 1,977 CANDIDATE objects are held out and ranked at the end.
- **Leakage removal:** dropped dispositions, `koi_score`, the four `koi_fpflag_*` flags, IDs; also excluded process signals (planet number, KOIs per star, missing-value flags). Keeping the leaky columns would give PR-AUC 0.998 instead of 0.955.
- **Features (12 base + 5 engineered):** transit/orbit, planet and star measurements, plus `depth_consistency`, `duration_ratio`, `implied_noise`, `star_density`, `rp_over_rs` (physical consistency checks). Log-transformed, median-imputed, standardised where needed.
- **Validation:** 20% test set and 5-fold CV on the rest, both **grouped by star**.
- **Models:** logistic regression, random forest, XGBoost, sklearn MLP, PyTorch residual MLP, XGBoost + ResMLP ensemble (primary, chosen on validation only).
- **Primary metric:** PR-AUC; also recall at 95% precision, F1, ROC-AUC, Brier.

## Final results
| Model (17 features) | Validation PR-AUC | **Test PR-AUC** (evaluated once) |
|---|---|---|
| Logistic regression | 0.736 | 0.767 |
| Random forest | 0.955 | 0.963 |
| XGBoost | 0.961 | 0.968 |
| PyTorch ResMLP | 0.959 | 0.963 |
| **Ensemble XGB + ResMLP (primary)** | **0.965** | **0.968** (95% CI 0.954-0.979) |

Failure mitigation: deferring the 7% least certain objects to human vetters halves the error rate on the rest (6.7% -> 3.9%, validation); threshold/calibration/re-weighting/specialist models did not help. The neural network and ensemble add at most +0.004 validation PR-AUC (nothing on test) at 5-37x the cost, so XGBoost is the recommended model.

Primary model on test: ROC-AUC 0.982, F1 0.926, recall 0.81 at 95.6% precision (threshold 0.80 fixed on validation), 87 errors on 1,546 objects. XGBoost alone ties the ensemble on test. TabPFN could not be run in our environment (gated weights); see `final/run_tabpfn_colab.py`.
Full report: `report/final_report.pdf` (source: `report/final_report.tex`). Ranked CANDIDATE list: `results/candidate_ranking.csv`.

## Repository structure
```
├── notebooks/kepler_project.ipynb   progress-report notebook (baseline experiments)
├── notebooks/kepler_final_notebook.ipynb  complete final experiment notebook
├── final/                           final experiments: common.py, s01..s05 scripts, torch_model.py, run_all.sh
├── report/                          final_report.tex / final_report.pdf
├── data/README.md                   how to download the dataset (kepler_cumulative.csv is included)
├── models/                          final_xgboost.joblib, final_resmlp.joblib (+ progress-report models) + MODEL_CARD.md
├── results/                         all metrics (CSV/JSON), figures, candidate ranking
├── predict.py                       score a CSV with the final ensemble
└── requirements.txt
```

## How to run

The reproduction was configured for Python 3.12. On macOS, XGBoost also needs the OpenMP runtime available to the Python process. Run these commands from the repository root.

```bash
pip install -r requirements.txt
bash final/run_all.sh                     # writes results/, figures and model files
python predict.py data/kepler_cumulative.csv
```
The main held-out scores use the frozen configuration in `results/frozen_config.json`. The split sensitivity analysis uses only the development partition. The final downloadable models are refitted on all labeled rows after the main evaluation for candidate scoring.

For the notebook, open `notebooks/kepler_final_notebook.ipynb` in a Python kernel with these dependencies and run from the `notebooks/` directory.

## Reproduction results

The README table, the report's main results, and the downloadable models are from the original project run. The executed final notebook is a separate fresh-kernel rerun. It selected the same primary model family but a different XGBoost configuration. Its test average precision was 0.9680 for XGBoost and 0.9695 for the ensemble. Changed rerun files are saved under `results/reproduction/`. The development-only sensitivity tables are in `results/sensitivity_*.csv`.

The compact-feature and calibration-correction results are exploratory. The report describes the sources of dependence in those comparisons and how to obtain an independent estimate.

## Status
- [x] Proposal and initial baseline (Oct 7)
- [x] Progress report (Nov 6)
- [x] Final report, final models and results (Nov 20)
