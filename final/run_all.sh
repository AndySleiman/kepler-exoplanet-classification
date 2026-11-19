#!/usr/bin/env bash
# Re-runs every final experiment (about 25-30 min on 2 CPU cores). Run from inside final/.
set -e
python s01_tune_classical.py      # LR / RF / XGBoost / sklearn-MLP, base vs +new features (validation only)
python s02_torch.py               # PyTorch residual MLP (validation only)
python s03_analysis.py            # model selection, failure analysis, failure fixes, SHAP, complexity (validation only)
python s03b_data_leakage.py       # dataset statistics and leakage ablation (validation only)
python s04_test_and_candidates.py # freeze -> ONE test evaluation -> rank CANDIDATEs
python s05_figures.py
python s06_extra_mitigations.py   # deferral, weak-signal specialist, interpretability trade-off (validation only)
python s07_sensitivity.py         # sensitivity to the split / evaluation design (validation only, ~15 min)
# optional, needs Hugging Face access (gated weights): python run_tabpfn_colab.py
