# Kepler Exoplanet Classification

CMPS 261 (Machine Learning) course project, American University of Beirut — Track A.

**Team:** Joe Khoury, Andy Sleiman, Anthony Merhy

## Problem
NASA's Kepler telescope flagged ~9,500 "Kepler Objects of Interest" (KOIs): stars whose brightness dips in a way that could be a planet passing in front of them. Many are false positives (eclipsing binary stars, noise, light from a neighbouring star).
We classify each KOI as a real planet or a false positive using **only measurements available before human vetting**, so the model can help scientists rank which detections deserve follow-up.

## Approach
- **Labels:** CONFIRMED (1) vs FALSE POSITIVE (0); 7,585 objects. CANDIDATE objects are held out and scored later.
- **Leakage removal:** dropped dispositions, `koi_score`, the four `koi_fpflag_*` flags, IDs; also excluded process signals (KOIs per star, missing-value flags).
- **Features (12):** transit/orbit (period, duration, depth, impact, SNR), planet (radius, equilibrium temperature, insolation), star (temperature, log g, radius, Kepler magnitude). Log-transformed, median-imputed, standardised where needed.
- **Validation:** 20% held-out test set and 5-fold cross-validation, both **grouped by star** so no star appears on both sides.
- **Model ladder:** logistic regression → random forest → XGBoost (classical) → MLP neural network → XGBoost + MLP ensemble.
- **Primary metric:** PR-AUC; also recall at 95% precision, F1, ROC-AUC, Brier score.

## Results so far (validation, 5-fold grouped CV)
| Model | PR-AUC | Recall @ 95% precision | Brier |
|---|---|---|---|
| Logistic regression | 0.731 | 0.000 | 0.158 |
| Random forest | 0.945 | 0.669 | 0.063 |
| **XGBoost** | **0.957** | 0.748 | 0.056 |
| MLP | 0.952 | 0.739 | 0.058 |
| XGBoost + MLP ensemble | 0.960 | 0.787 | 0.052 |
| *XGBoost with leaky columns (not valid)* | *0.998* | | |

## Repository structure
```
├── notebooks/kepler_project.ipynb   all code: cleaning, preprocessing, tuning, models, failure analysis
├── data/README.md                   how to download the dataset
├── models/                          trained models (download these) + MODEL_CARD.md
├── results/                         validation metrics and figures
├── predict.py                       score a CSV with the trained model
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
# 1. download the data (see data/README.md) to data/kepler_cumulative.csv
# 2. run everything
jupyter notebook notebooks/kepler_project.ipynb      # Run All, ~15 min on CPU
# 3. use the trained model
python predict.py data/kepler_cumulative.csv
```

## Trained model
Download `models/xgboost_pipeline.joblib` (main model) or `models/mlp_pipeline.joblib`. See `models/MODEL_CARD.md`.

## Status
- [x] Proposal and initial baseline (Oct 7)
- [x] Progress report (Nov 6)
- [ ] Final report, poster and final models (Nov 20)
