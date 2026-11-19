# Model card

**Final model:** equal-weight average of `final_xgboost.joblib` (XGBoost, 200 trees, depth 6) and `final_resmlp.joblib` (PyTorch residual MLP, 5-seed ensemble, 48,965 weights). Both are full scikit-learn Pipelines (log10 + engineered features -> median imputation -> [scaling] -> model) refit on all 7,587 labelled objects with hyper-parameters frozen on validation data.

**Task:** CONFIRMED (1) vs FALSE POSITIVE (0) Kepler Objects of Interest.
**Inputs:** the 12 columns `koi_period, koi_duration, koi_depth, koi_impact, koi_model_snr, koi_prad, koi_teq, koi_insol, koi_steff, koi_slogg, koi_srad, koi_kepmag`; five engineered features are computed inside the pipeline. No vetting outputs (`koi_score`, `koi_fpflag_*`, dispositions) are used.

**Performance (ensemble trained on the 80% train part, single evaluation on the held-out 20% test set, grouped by star):** PR-AUC 0.968 (95% CI 0.954-0.979), ROC-AUC 0.982, F1 0.926 at threshold 0.5. Thresholds chosen on validation: 0.40 (F1-optimal), 0.80 (95% precision; gave 95.6% precision and 81% recall on test).

**Use:** `python predict.py data/kepler_cumulative.csv` (needs `final/common.py` and `final/torch_model.py`, which `predict.py` adds to the path).

**Known weaknesses:** weak signals (SNR 10-20), Earth-sized planets, rare giant/grazing/hot-star planets (usually predicted as false positives), objects with missing measurements (almost all false positives in the catalogue), and false positives that closely mimic planets. Candidate scores are a prioritisation aid, not confirmation.

`xgboost_pipeline.joblib` and `mlp_pipeline.joblib` are the progress-report models (12 features, previous data snapshot) kept for reference.
