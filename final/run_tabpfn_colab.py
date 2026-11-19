"""TabPFN experiment -- run this on Google Colab (or any machine that can reach Hugging Face).

TabPFN's pre-trained weights are gated: accept the terms at https://huggingface.co/Prior-Labs/tabpfn_3_5 ,
create an HF token and set it as the HF_TOKEN environment variable (Colab: Secrets tab).

    pip install tabpfn scikit-learn pandas numpy
    python run_tabpfn_colab.py            # expects ../data/kepler_cumulative.csv, run from the final/ folder

It uses EXACTLY the same split / folds / features as the other experiments (common.py) and prints the
same metrics, so the row can be pasted into the results table. It only touches the train/validation
part (5-fold grouped CV); it never looks at the held-out test set.
"""
import time, json
import numpy as np
from common import *
from sklearn.model_selection import cross_val_predict
from tabpfn import TabPFNClassifier

df, lab, cand, tr, te, y, splits = load()
for engineered in (False, True):
    pipe = Pipeline(preprocessing(False, engineered) + [("m", TabPFNClassifier(random_state=RS))])
    t0 = time.time()
    p = cross_val_predict(pipe, tr[FEATS], y, cv=splits, method="predict_proba")[:, 1]
    el = time.time() - t0
    m = all_metrics(y, p); m["cv_seconds"] = el
    tag = "TabPFN +new features" if engineered else "TabPFN"
    print(tag, json.dumps({k: round(v, 4) for k, v in m.items()}))
    np.save(f"cache/oof_{tag.replace(' ', '_')}.npy", p)
