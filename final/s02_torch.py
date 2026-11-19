"""Step 2: tune the PyTorch residual MLP with the same grouped 5-fold CV (random search, 10 configs)."""
import sys, pickle, time, itertools
sys.path.insert(0, ".")
from common import *
from torch_model import TorchMLP
from sklearn.model_selection import cross_val_predict, ParameterSampler

df, lab, cand, tr, te, y, splits = load()
space = {"m__width": [64, 128, 256], "m__n_blocks": [1, 2, 3], "m__dropout": [0.0, 0.1, 0.2, 0.3],
         "m__lr": [1e-3, 3e-3, 1e-2], "m__weight_decay": [1e-4, 1e-3, 1e-2, 1e-1], "m__epochs": [40, 80]}
out = {}
for engineered in (True,):
    tag = "PyTorch ResMLP" + (" +new features" if engineered else "")
    best = None; t_tune = time.time(); log = []
    for i, prm in enumerate(ParameterSampler(space, n_iter=10, random_state=RS)):
        pipe = Pipeline(preprocessing(True, engineered) + [("m", TorchMLP(n_seeds=1, random_state=RS))]).set_params(**prm)
        p = cross_val_predict(pipe, tr[FEATS], y, cv=splits, method="predict_proba")[:, 1]
        ap = average_precision_score(y, p); log.append((prm, ap))
        print(tag, i, round(ap, 4), prm, flush=True)
        if best is None or ap > best[0]: best = (ap, prm)
    tune = time.time() - t_tune
    # final: 5-seed ensemble with the best hyperparameters
    final = Pipeline(preprocessing(True, engineered) + [("m", TorchMLP(n_seeds=5, random_state=RS))]).set_params(**best[1])
    p = cross_val_predict(final, tr[FEATS], y, cv=splits, method="predict_proba")[:, 1]
    t0 = time.time(); final.fit(tr[FEATS], y); fit = time.time() - t0
    m = all_metrics(y, p)
    m["PR-AUC fold SD"] = float(np.std([average_precision_score(y[v], p[v]) for _, v in splits]))
    m.update({"Configs tried": 10, "Tuning time (s)": tune, "Fit time (s)": fit})
    print(f"{tag}: PR-AUC {m['PR-AUC']:.4f} R@P95 {m['Recall@P95']:.3f} Brier {m['Brier']:.4f} fit {fit:.0f}s", flush=True)
    out[tag] = dict(name=tag, metrics=m, oof=p, model=final, params=best[1], search_log=log)
    pickle.dump(out, open("cache/s02_torch.pkl", "wb"))
print("DONE")
