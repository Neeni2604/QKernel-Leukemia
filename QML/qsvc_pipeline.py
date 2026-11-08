"""qsvc_pipeline.py - Quantum SVC pipeline for QKernel Leukemia.

Same CV harness and gene selection as baselines/baselines.py, so results are directly
comparable. Every fold does, using TRAINING patients only:

    Welch t gene selection (top k) -> scale to a fixed range -> quantum kernel -> SVC

Two ways to compute the quantum kernel K(x, y) = |<psi(x)|psi(y)>|^2:

  --kernel exact    (default) exact statevectors, no shot noise, deterministic, fast.
                    Use this for all the accuracy / qubit / feature-map sweeps.
  --kernel sampler  FidelityQuantumKernel + StatevectorSampler. This SAMPLES shots
                    (default 1024), so kernel entries are noisy. Use it for the
                    finite-shot study. (StatevectorSampler is NOT exact: it draws
                    measurement shots from the exact state.)

The encoding scale matters a lot. Features become rotation angles, so their spread
sets the kernel "bandwidth". Too wide and the kernel turns into nearly the identity
matrix (every patient looks different from every other), and the SVM can only
predict the majority class. --scaler minmax maps genes to [0, 1] on the training
fold, and --scale multiplies that (use --scale 3.14159 for [0, pi]).
Sweep --scales and report the whole curve rather than only the best value.

Usage:
    python qsvc_pipeline.py                                   # k=8, minmax [0,1], 5x5 CV
    python qsvc_pipeline.py --ks 2 4 6 8 --scales 0.5 1 3.14159
    python qsvc_pipeline.py --ks 4 --kernel sampler --shots 256 --folds 3 --repeats 1
    python qsvc_pipeline.py --feature-map z                   # no entanglement (classical-simulable)
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "baselines"))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qiskit.quantum_info import Statevector

try:
    from qiskit.circuit.library import z_feature_map, zz_feature_map
except ImportError:  # older Qiskit versions only have the classes
    from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap

    def z_feature_map(feature_dimension, reps=2):
        return ZFeatureMap(feature_dimension, reps=reps)

    def zz_feature_map(feature_dimension, reps=2, entanglement="full"):
        return ZZFeatureMap(feature_dimension, reps=reps, entanglement=entanglement)

from data_loader import clip_and_log, load_golub, validate
from selection import WelchTSelector

SCORING = {
    "accuracy": "accuracy",
    "auroc": "roc_auc",
    "sensitivity": make_scorer(recall_score, pos_label=1),
    "specificity": make_scorer(recall_score, pos_label=0),
}

# Progress counter shared across cloned estimators (sklearn clones the model for each fold).
_PROGRESS = {"fold": 0, "total": 0}


def log(msg: str = "", end: str = "\n") -> None:
    print(msg, end=end, flush=True)


def _build_feature_map(name: str, k: int, reps: int = 1):
    if name == "zz":
        return zz_feature_map(feature_dimension=k, reps=reps, entanglement="linear")
    if name == "z":
        return z_feature_map(feature_dimension=k, reps=reps)
    raise ValueError(f"Unknown feature map '{name}'. Choose 'zz' or 'z'.")


class QuantumKernelPipeline(BaseEstimator, ClassifierMixin):
    """Fold-safe: Welch selection -> scaling -> quantum kernel -> SVC(precomputed)."""

    def __init__(self, k=8, feature_map="zz", reps=1, C=1.0, scaler="minmax",
                 scale=1.0, kernel="exact", shots=1024, seed=0, verbose=False,
                 class_weight=None):
        self.verbose = verbose
        self.class_weight = class_weight
        self.k = k
        self.feature_map = feature_map
        self.reps = reps
        self.C = C
        self.scaler = scaler
        self.scale = scale
        self.kernel = kernel
        self.shots = shots
        self.seed = seed

    # ---- kernel -------------------------------------------------------------
    def _states(self, Z):
        return np.array([Statevector(self._fm_.assign_parameters(z)).data for z in Z])

    def _kernel(self, A, B=None):
        if self.kernel == "exact":
            SA = self._states(A)
            SB = SA if B is None else self._states(B)
            return np.abs(SA.conj() @ SB.T) ** 2
        if self.kernel == "sampler":
            return self._sampler_kernel_.evaluate(A, B)
        raise ValueError(f"Unknown kernel '{self.kernel}'. Choose 'exact' or 'sampler'.")

    def _make_sampler_kernel(self):
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        sampler = StatevectorSampler(default_shots=self.shots, seed=self.seed)
        return FidelityQuantumKernel(
            feature_map=self._fm_, fidelity=ComputeUncompute(sampler=sampler)
        )

    # ---- preprocessing ------------------------------------------------------
    def _transform(self, X):
        Z = self.scaler_.transform(self.selector_.transform(np.asarray(X, dtype=float)))
        return Z * self.scale

    # ---- sklearn API --------------------------------------------------------
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        _PROGRESS["fold"] += 1
        t0 = time.perf_counter()
        if self.verbose:
            log(f"    fold {_PROGRESS['fold']}/{_PROGRESS['total']}: "
                f"{len(X)} train patients | selecting {self.k} genes (Welch t)... ", end="")
        self.selector_ = WelchTSelector(k=self.k).fit(X, y)
        Z = self.selector_.transform(X)
        self.scaler_ = (MinMaxScaler() if self.scaler == "minmax" else StandardScaler()).fit(Z)
        self.X_train_ = self._transform(X)
        self._fm_ = _build_feature_map(self.feature_map, self.k, self.reps)
        if self.kernel == "sampler":
            self._sampler_kernel_ = self._make_sampler_kernel()
        self._cache_ = None
        if self.verbose:
            log(f"done | training kernel {len(X)}x{len(X)}... ", end="")
        K_train = self._kernel(self.X_train_)
        self.svc_ = SVC(
            kernel="precomputed", C=self.C, class_weight=self.class_weight
        ).fit(K_train, y)
        self.classes_ = self.svc_.classes_
        if self.verbose:
            log(f"done, SVC fitted ({time.perf_counter() - t0:.1f}s)")
        return self

    def _test_kernel(self, X):
        """Kernel between X and the training set; cached because sklearn scorers call
        predict() and decision_function() separately on the same X."""
        X = np.asarray(X, dtype=float)
        key = (X.shape, hash(X.tobytes()))
        if self._cache_ is not None and self._cache_[0] == key:
            return self._cache_[1]
        t0 = time.perf_counter()
        K = self._kernel(self._transform(X), self.X_train_)
        self._cache_ = (key, K)
        if self.verbose:
            log(f"      scored {len(X)} test patients: kernel {K.shape[0]}x{K.shape[1]} "
                f"({time.perf_counter() - t0:.1f}s)")
        return K

    def predict(self, X):
        return self.svc_.predict(self._test_kernel(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._test_kernel(X))


def evaluate(X, y, model, n_splits=5, n_repeats=5, seed=0) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING, error_score="raise")
    out = {}
    for m in SCORING:
        vals = res[f"test_{m}"]
        out[f"{m}_mean"], out[f"{m}_std"] = float(vals.mean()), float(vals.std())
    return out


def _main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ap = argparse.ArgumentParser(description="Quantum SVC pipeline for Golub leukemia data.")
    ap.add_argument("--data-dir", default=str(HERE.parent / "data"))
    ap.add_argument("--ks", "--k", dest="ks", type=int, nargs="+", default=[8],
                    help="Number of genes (= qubits) to select per fold; several values sweep")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0],
                    help="Multiplier applied after scaling (angle spread); several values sweep")
    ap.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    ap.add_argument("--feature-map", choices=["zz", "z"], default="zz")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--Cs", "--C", dest="Cs", type=float, nargs="+", default=[1.0],
                    help="SVC regularisation C; several values sweep (bigger = fits training data harder)")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="'balanced' stops the SVM from favouring the majority class (ALL)")
    ap.add_argument("--kernel", choices=["exact", "sampler"], default="exact")
    ap.add_argument("--shots", type=int, default=1024, help="Shots per circuit (sampler kernel)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/qsvc_results.csv")
    ap.add_argument("--quiet", action="store_true", help="Only print one line per config")
    args = ap.parse_args()

    t_start = time.perf_counter()
    log(f"[1/4] Loading data from {args.data_dir} ...")
    d = load_golub(args.data_dir)
    validate(d)
    X = clip_and_log(d.X).to_numpy()
    y = d.y.to_numpy()
    log(f"      {X.shape[0]} patients x {X.shape[1]} genes loaded and validated; "
        f"clipped + log2 transformed (ALL={int((y == 0).sum())}, AML={int((y == 1).sum())})")

    log(f"[2/4] Setup: {args.folds}-fold x {args.repeats} repeated CV; "
        f"{args.feature_map.upper()} map reps={args.reps}, scaler={args.scaler}, "
        f"class_weight={args.class_weight}, kernel={args.kernel}" + (f" ({args.shots} shots)" if args.kernel == "sampler" else ""))

    sweep = HERE.parent / "baselines" / "results" / "k_sweep.csv"
    classical = pd.read_csv(sweep) if sweep.exists() else None
    log("      classical k_sweep.csv " + ("found: will print best classical per k" if classical is not None
                                         else "not found: skipping classical comparison"))

    configs = list(itertools.product(args.ks, args.scales, args.Cs))
    class_weight = None if args.class_weight == "none" else "balanced"
    total_folds = args.folds * args.repeats
    log(f"[3/4] Running {len(configs)} config(s) x {total_folds} folds ...")

    rows = []
    for i, (k, scale, C) in enumerate(configs, start=1):
        log(f"  config {i}/{len(configs)}: k={k} genes/qubits, scale={scale}, C={C}")
        _PROGRESS["fold"], _PROGRESS["total"] = 0, total_folds
        t_cfg = time.perf_counter()
        model = QuantumKernelPipeline(
            k=k, feature_map=args.feature_map, reps=args.reps, C=C,
            scaler=args.scaler, scale=scale, kernel=args.kernel,
            shots=args.shots, seed=args.seed, verbose=not args.quiet,
            class_weight=class_weight,
        )
        r = evaluate(X, y, model, args.folds, args.repeats, args.seed)
        rows.append({
            "model": f"qsvc_{args.feature_map}_{args.kernel}", "k_genes": k,
            "scaler": args.scaler, "scale": scale, "reps": args.reps, "C": C,
            "class_weight": args.class_weight,
            "shots": args.shots if args.kernel == "sampler" else 0, **r,
        })
        line = (f"  -> k={k:<3d} scale={scale:<7.3f} C={C:<6g} acc {r['accuracy_mean']:.3f} +/- "
                f"{r['accuracy_std']:.3f}  auroc {r['auroc_mean']:.3f}  "
                f"sens {r['sensitivity_mean']:.3f}  spec {r['specificity_mean']:.3f}  "
                f"({time.perf_counter() - t_cfg:.1f}s)")
        if classical is not None:
            c = classical[(classical["k_genes"] == k) & (classical["model"] != "majority_class")]
            if not c.empty:
                best = c.loc[c["accuracy_mean"].idxmax()]
                line += f"  | best classical {best['model']} {best['accuracy_mean']:.3f}"
        print(line)

    log("[4/4] Saving results ...")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"      Saved: {out}  (total time {time.perf_counter() - t_start:.1f}s)")


if __name__ == "__main__":
    _main()