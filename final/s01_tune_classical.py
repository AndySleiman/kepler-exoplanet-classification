"""Step 1: re-run the model ladder on the current data (base 12 features vs 12 + 5 engineered features).
All tuning uses ONLY the 80% train/validation part (5-fold grouped CV). The test set is not touched."""
import sys, pickle, time
sys.path.insert(0, ".")
from common import *
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

df, lab, cand, tr, te, y, splits = load()
BASE12 = FEATS

def run(name, pipe, space, n_iter, engineered):
    X = tr[BASE12]
    if n_iter is None:
        s = GridSearchCV(pipe, space, scoring="average_precision", cv=splits, n_jobs=2)
    else:
        s = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision", cv=splits, n_jobs=2, random_state=RS)
    t0 = time.time(); s.fit(X, y); tune = time.time() - t0
    best = s.best_estimator_
    p = cross_val_predict(best, X, y, cv=splits, method="predict_proba")[:, 1]
    t0 = time.time(); best.fit(X, y); fit = time.time() - t0
    m = all_metrics(y, p)
    m["PR-AUC fold SD"] = float(np.std([average_precision_score(y[v], p[v]) for _, v in splits]))
    m.update({"Configs tried": len(s.cv_results_["params"]), "Tuning time (s)": tune, "Fit time (s)": fit})
    print(f"{name:34s} PR-AUC {m['PR-AUC']:.4f}  R@P95 {m['Recall@P95']:.3f}  Brier {m['Brier']:.4f}  tune {tune:.0f}s", flush=True)
    return dict(name=name, metrics=m, oof=p, model=best, params=s.best_params_)

LR = lambda: LogisticRegression(max_iter=5000, class_weight="balanced")
RF = lambda: RandomForestClassifier(n_jobs=1, random_state=RS)
XG = lambda: XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=RS, tree_method="hist")
ML = lambda: MLPClassifier(max_iter=600, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=RS)
spaces = {
 "Logistic regression": (LR, True, {"m__C": [0.001, 0.01, 0.1, 1, 10, 100], "m__penalty": ["l2"]}, None),
 "Random forest": (RF, False, {"m__n_estimators": [300, 600], "m__max_depth": [None, 10, 20], "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": ["sqrt", 0.5]}, 12),
 "XGBoost": (XG, False, {"m__n_estimators": [200, 400, 800], "m__max_depth": [3, 4, 6, 8], "m__learning_rate": [0.02, 0.05, 0.1], "m__subsample": [0.7, 0.85, 1.0], "m__colsample_bytree": [0.6, 0.8, 1.0], "m__min_child_weight": [1, 3, 5], "m__reg_lambda": [0.5, 1, 5]}, 40),
 "MLP (sklearn)": (ML, True, {"m__hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64), (64, 64)], "m__alpha": [1e-4, 1e-3, 1e-2, 1e-1], "m__learning_rate_init": [1e-3, 3e-3], "m__batch_size": [64, 256]}, 20),
}
out = {}
for engineered in (False, True):
    for name, (mk, scale, space, n_iter) in spaces.items():
        tag = name + (" +new features" if engineered else "")
        out[tag] = run(tag, Pipeline(preprocessing(scale, engineered) + [("m", mk())]), space, n_iter, engineered)
        pickle.dump(out, open("cache/s01_classical.pkl", "wb"))
print("DONE")
