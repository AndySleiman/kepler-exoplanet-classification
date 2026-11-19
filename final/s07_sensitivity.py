"""Step 7: how sensitive are the conclusions to the split and the evaluation design?
(A) 5 repeats of grouped 5-fold CV with different fold seeds (train/validation part only);
(B) 10 independent grouped 80/20 splits of ALL labelled data, frozen hyper-parameters, models refit on each split;
(C) grouped vs. ordinary (ungrouped) cross-validation.
Hyper-parameters are the ones frozen in s01/s02 (no re-tuning). The ORIGINAL held-out test set is not used for any decision here."""
import sys, os, pickle
sys.path.insert(0, ".")
from common import *
from torch_model import TorchMLP
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit, StratifiedGroupKFold

QUICK = os.environ.get("QUICK") == "1"
df, lab, cand, tr, te, y, splits = load()
S1 = pickle.load(open("cache/s01_classical.pkl", "rb")); S2 = pickle.load(open("cache/s02_torch.pkl", "rb"))
M = {"Logistic regression": clone(S1["Logistic regression +new features"]["model"]), "Random forest": clone(S1["Random forest +new features"]["model"]),
     "XGBoost": clone(S1["XGBoost +new features"]["model"]), "ResMLP": clone(S2["PyTorch ResMLP +new features"]["model"])}
def fit_pred(name, Xa, ya, Xb):
    return clone(M[name]).fit(Xa, ya).predict_proba(Xb)[:, 1]
def metrics(yt, P):
    P = dict(P); P["Ensemble XGB+ResMLP"] = (P["XGBoost"] + P["ResMLP"]) / 2
    return {k: (average_precision_score(yt, p), recall_at_precision(yt, p)) for k, p in P.items()}

# ---------------- (A) repeated grouped CV on the train/validation part ----------------
reps = 1 if QUICK else 5; rowsA = []
for r in range(reps):
    cvr = list(StratifiedGroupKFold(5, shuffle=True, random_state=100 + r).split(tr, y, tr["kepid"]))
    oof = {k: np.zeros(len(y)) for k in M}
    for a, b in cvr:
        for k in M: oof[k][b] = fit_pred(k, tr[FEATS].iloc[a], y[a], tr[FEATS].iloc[b])
    for k, (ap, r95) in metrics(y, oof).items(): rowsA.append({"repeat": r, "model": k, "PR-AUC": ap, "Recall@P95": r95})
    print("repeat", r, "done", flush=True)
A = pd.DataFrame(rowsA); A.to_csv("../results/sensitivity_repeated_cv_raw.csv", index=False)
summA = A.groupby("model")[["PR-AUC", "Recall@P95"]].agg(["mean", "std"]); print("\n(A) repeated grouped 5-fold CV\n", summA.round(4).to_string()); summA.to_csv("../results/sensitivity_repeated_cv.csv")
pv = A.pivot(index="repeat", columns="model", values="PR-AUC"); d = pv["Ensemble XGB+ResMLP"] - pv["XGBoost"]
print("Ensemble - XGBoost PR-AUC per repeat:", d.round(4).tolist(), "| mean", round(d.mean(), 4), "| ensemble better in", int((d > 0).sum()), "of", len(d))

# ---------------- (B) many independent outer splits (frozen hyper-parameters) ----------------
nout = 2 if QUICK else 10; rowsB = []
for s in range(nout):
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=1000 + s).split(lab, lab["y"], lab["kepid"]))
    la, lb = lab.iloc[a], lab.iloc[b]
    P = {k: fit_pred(k, la[FEATS], la["y"].values, lb[FEATS]) for k in M}
    for k, (ap, r95) in metrics(lb["y"].values, P).items(): rowsB.append({"split": s, "model": k, "PR-AUC": ap, "Recall@P95": r95})
    print("outer split", s, "done", flush=True)
B = pd.DataFrame(rowsB); B.to_csv("../results/sensitivity_outer_splits_raw.csv", index=False)
summB = B.groupby("model")[["PR-AUC", "Recall@P95"]].agg(["mean", "std", "min", "max"]); print("\n(B) 10 independent grouped 80/20 splits\n", summB.round(4).to_string()); summB.to_csv("../results/sensitivity_outer_splits.csv")
pvB = B.pivot(index="split", columns="model", values="PR-AUC"); dB = pvB["Ensemble XGB+ResMLP"] - pvB["XGBoost"]
print("Ensemble - XGBoost test PR-AUC per split:", dB.round(4).tolist(), "| mean", round(dB.mean(), 4), "| ensemble better in", int((dB > 0).sum()), "of", len(dB))

# ---------------- (C) grouped vs ungrouped CV (XGBoost, same hyper-parameters) ----------------
ung = np.zeros(len(y)); grp = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=RS).split(tr, y):
    ung[b] = fit_pred("XGBoost", tr[FEATS].iloc[a], y[a], tr[FEATS].iloc[b])
for a, b in splits: grp[b] = fit_pred("XGBoost", tr[FEATS].iloc[a], y[a], tr[FEATS].iloc[b])
C = pd.DataFrame({"ungrouped 5-fold CV": [average_precision_score(y, ung), recall_at_precision(y, ung)], "grouped by star 5-fold CV": [average_precision_score(y, grp), recall_at_precision(y, grp)]}, index=["PR-AUC", "Recall@P95"])
print("\n(C) grouped vs ungrouped\n", C.round(4).to_string()); C.to_csv("../results/sensitivity_grouping.csv")
