# Model card

| | XGBoost (main model) | MLP |
|---|---|---|
| File | `xgboost_pipeline.joblib` | `mlp_pipeline.joblib` |
| Validation PR-AUC (5-fold grouped CV) | 0.957 ± 0.003 | 0.952 ± 0.003 |
| Recall at 95% precision | 0.748 | 0.739 |
| Size | 400 trees, depth 8 | 3 hidden layers (256, 128, 64), 44,545 parameters |

**Task:** CONFIRMED (1) vs FALSE POSITIVE (0) Kepler Objects of Interest.

**Inputs (12 features):** `koi_period, koi_duration, koi_depth, koi_impact, koi_model_snr, koi_prad, koi_teq, koi_insol, koi_steff, koi_slogg, koi_srad, koi_kepmag`.
No vetting outputs (`koi_score`, `koi_fpflag_*`, dispositions) are used.

**Each file is a full scikit-learn Pipeline** (log10 transform → median imputation → [scaling] → model), so it takes raw table columns.

**Load it:**
```python
import joblib
from predict import log_transform   # needed so the saved pipeline can find its transform
model = joblib.load("models/xgboost_pipeline.joblib")
probs = model.predict_proba(df[FEATS])[:, 1]
```
Or simply: `python predict.py data/kepler_cumulative.csv`

**Known weaknesses:** low-SNR signals (SNR 10–20), small false positives that look like planets, and rare-planet regions (radius > 15 Earth radii, grazing transits, hot stars, rows with missing values). See the notebook, Section 7.

**Status:** progress-report version (Nov 6). Not yet evaluated on the held-out test set.
