"""baselines.py - fold-safe classical baselines + leakage check for QKernel Leukemia.

Every fold does this on the TRAINING patients only:
    Welch t gene selection (top k) -> standardize -> classifier

The same harness will later take the quantum models (QSVC etc.): swap the
classifier, keep everything else identical.

Usage:
    python baselines.py                            # k = 2,4,6,8; 5x5 repeated CV
    python baselines.py --ks 4 --leakage-check     # also run the shuffled-label test
    python baselines.py --data-dir data --out results/classical_baselines.csv

Sensitivity = fraction of AML patients caught; specificity = fraction of ALL cleared.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_loader import clip_and_log, load_golub, validate
from selection import WelchTSelector, welch_t_scores

SCORING = {
    "accuracy": "accuracy",
    "auroc": "roc_auc",
    "sensitivity": make_scorer(recall_score, pos_label=1),
    "specificity": make_scorer(recall_score, pos_label=0),
}

MODELS = {
    "majority_class": lambda seed: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000),
    "svm_linear": lambda seed: SVC(kernel="linear"),
    "svm_rbf": lambda seed: SVC(kernel="rbf"),
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
}


def make_pipeline(model, k: int) -> Pipeline:
    return Pipeline(
        [("select", WelchTSelector(k=k)), ("scale", StandardScaler()), ("clf", model)]
    )


def make_cv(n_splits: int = 5, n_repeats: int = 5, seed: int = 0):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def evaluate(X, y, model, k, n_splits=5, n_repeats=5, seed=0) -> dict:
    """Mean and std of each metric across all CV folds."""
    res = cross_validate(
        make_pipeline(model, k),
        np.asarray(X),
        np.asarray(y),
        cv=make_cv(n_splits, n_repeats, seed),
        scoring=SCORING,
        error_score="raise",
    )
    out = {}
    for m in SCORING:
        vals = res[f"test_{m}"]
        out[f"{m}_mean"], out[f"{m}_std"] = float(vals.mean()), float(vals.std())
    return out


def run_all(X, y, ks, n_splits=5, n_repeats=5, seed=0) -> pd.DataFrame:
    rows = []
    for name, factory in MODELS.items():
        for k in ks:
            r = evaluate(X, y, factory(seed), k, n_splits, n_repeats, seed)
            rows.append({"model": name, "k_genes": k, **r})
            print(f"  {name:20s} k={k}: acc {r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f}")
    return pd.DataFrame(rows)


def leakage_check(X, y, k=4, n_splits=5, n_repeats=5, seed=0) -> dict:
    """Shuffle labels, then compare a CORRECT pipeline with a deliberately LEAKY one.

    Labels are random, so honest accuracy should sit near the majority baseline.
    The leaky version picks genes on ALL patients first, then cross-validates, and
    scores much higher on pure noise. That gap is exactly what leakage looks like.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(np.asarray(y))
    cv = make_cv(n_splits, n_repeats, seed)

    proper = cross_validate(
        make_pipeline(SVC(kernel="rbf"), k), X, y_shuf, cv=cv, scoring="accuracy"
    )["test_score"]

    idx = np.argsort(-welch_t_scores(X, y_shuf))[:k]  # selection sees ALL labels
    leaky = cross_validate(
        Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
        X[:, idx], y_shuf, cv=cv, scoring="accuracy",
    )["test_score"]

    return {
        "majority_baseline": float(max(y_shuf.mean(), 1 - y_shuf.mean())),
        "proper_shuffled_acc": float(proper.mean()),
        "leaky_shuffled_acc": float(leaky.mean()),
    }


def _main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).parent.parent / "data"))
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/classical_baselines.csv")
    ap.add_argument("--leakage-check", action="store_true")
    args = ap.parse_args()

    d = load_golub(args.data_dir)
    validate(d)
    X, y = clip_and_log(d.X).to_numpy(), d.y.to_numpy()

    print(f"Running {args.folds}-fold x {args.repeats} repeated CV on {X.shape[0]} patients")
    df = run_all(X, y, args.ks, args.folds, args.repeats, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nSaved: {out}")

    if args.leakage_check:
        print("\nLeakage check (labels shuffled, k={}):".format(args.ks[0]))
        for key, val in leakage_check(X, y, args.ks[0], args.folds, args.repeats, args.seed).items():
            print(f"  {key}: {val:.3f}")
        print("  Expect proper at or below the majority baseline; leaky clearly higher.")


if __name__ == "__main__":
    _main()