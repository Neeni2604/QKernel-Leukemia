"""qsvc_pipeline.py - Quantum kernel SVM for QKernel Leukemia, following the IBM
"Quantum kernel methods" tutorial as closely as possible.

Tutorial method (what --kernel page / page-noisy do):
  1. Encode each patient x in a feature-map circuit (z_feature_map / zz_feature_map).
  2. For every pair (x_i, x_j) build  unitary_overlap(U(x_i), U(x_j))  and measure_all().
  3. (page-noisy only) transpile with generate_preset_pass_manager(optimization_level=3)
     for an IBM device (default FakeBrisbane, the device family the tutorial used).
  4. Run it on a sampler. The kernel entry is the probability of the all-zeros outcome,
     K_ij = |<psi(x_j)|psi(x_i)>|^2 = counts['00..0'] / shots.
  5. Gram matrix: one circuit per pair, only the upper triangle is run, diagonal = 1.
     Test matrix: one circuit per (test, train) pair.
  6. SVC(kernel="precomputed"): fit(K_train, y_train), predict(K_test).

Kernel modes:
  exact        exact statevector overlaps. No circuits, no shots. Fast: use for sweeps.
  page         steps 1-2, 4-6 on StatevectorSampler (the tutorial's simulator option).
               Shot noise only (--shots). No transpilation.
  page-noisy   steps 1-6 on AerSimulator.from_backend(FakeBrisbane): transpiled circuits
               plus a hardware-like noise model. Slow, and expect it to hurt: the tutorial
               itself reports the ZZ map collapsing on hardware because of depth and noise.

Deliberate differences from the tutorial: genes are chosen by a Welch t-test inside every
CV fold and rescaled (--scaler minmax, --scale) before encoding, evaluation is repeated
stratified CV, and there is no real-hardware run (that is a two-line change to the
sampler once you have an IBM Quantum account).

Usage:
    python qsvc_pipeline.py --ks 4 8 --scales 0.1 0.25 0.5 1 --quiet        # exact, fast
    python qsvc_pipeline.py --ks 4 --kernel page --folds 3 --repeats 1      # tutorial method
    python qsvc_pipeline.py --ks 4 --kernel page-noisy --folds 3 --repeats 1 --scales 0.25
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

from qiskit.circuit.library import unitary_overlap
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
_BACKEND_CACHE: dict = {}


def log(msg: str = "", end: str = "\n") -> None:
    print(msg, end=end, flush=True)


def _build_feature_map(name: str, k: int, reps: int = 1):
    if name == "zz":
        return zz_feature_map(feature_dimension=k, reps=reps, entanglement="linear")
    if name == "z":
        return z_feature_map(feature_dimension=k, reps=reps)
    raise ValueError(f"Unknown feature map '{name}'. Choose 'zz' or 'z'.")


def _noisy_backend_and_pm(fake_name: str, opt_level: int):
    """AerSimulator with the noise model + coupling map of an IBM device, and the preset
    pass manager the tutorial uses. Cached so it is built once per run."""
    key = (fake_name, opt_level)
    if key not in _BACKEND_CACHE:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import fake_provider

        backend = AerSimulator.from_backend(getattr(fake_provider, fake_name)())
        pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
        _BACKEND_CACHE[key] = (backend, pm)
    return _BACKEND_CACHE[key]


class OverlapKernel:
    """Kernel exactly as in the tutorial: unitary_overlap circuits, P(all zeros)."""

    def __init__(self, feature_map, mode="page", shots=1024, seed=0,
                 fake_backend="FakeBrisbane", opt_level=3, verbose=False):
        self.fm, self.mode, self.shots, self.seed = feature_map, mode, shots, seed
        self.fake_backend, self.opt_level, self.verbose = fake_backend, opt_level, verbose

    def _p_zero(self, circuits) -> np.ndarray:
        n = self.fm.num_qubits
        zeros = "0" * n
        if self.mode == "page":
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(default_shots=self.shots, seed=self.seed)
            result = sampler.run([(c,) for c in circuits]).result()
            counts = [result[i].data.meas.get_counts() for i in range(len(circuits))]
        else:  # page-noisy
            backend, pm = _noisy_backend_and_pm(self.fake_backend, self.opt_level)
            isa = pm.run(circuits)
            counts = backend.run(isa, shots=self.shots, seed_simulator=self.seed).result().get_counts()
            counts = [counts] if isinstance(counts, dict) else counts
        return np.array([c.get(zeros, 0) / self.shots for c in counts])

    def evaluate(self, A, B=None) -> np.ndarray:
        symmetric = B is None
        B = A if symmetric else B
        circ_A = [self.fm.assign_parameters(a) for a in A]
        circ_B = circ_A if symmetric else [self.fm.assign_parameters(b) for b in B]
        # symmetric Gram matrix: only the upper triangle is run; the diagonal is set to 1
        pairs = [(i, j) for i in range(len(A)) for j in range(i + 1 if symmetric else 0, len(B))]
        circuits = []
        for i, j in pairs:
            c = unitary_overlap(circ_A[i], circ_B[j])
            c.measure_all()
            circuits.append(c)
        if self.verbose:
            log(f"[{len(circuits)} overlap circuits, {self.shots} shots each] ", end="")
        p = self._p_zero(circuits) if circuits else np.array([])
        K = np.zeros((len(A), len(B)))
        for (i, j), pij in zip(pairs, p):
            K[i, j] = pij
            if symmetric:
                K[j, i] = pij
        if symmetric:
            np.fill_diagonal(K, 1.0)
        return K


class QuantumKernelPipeline(BaseEstimator, ClassifierMixin):
    """Fold-safe: Welch selection -> scaling -> quantum kernel -> SVC(precomputed)."""

    def __init__(self, k=8, feature_map="zz", reps=1, C=1.0, scaler="minmax",
                 scale=1.0, kernel="exact", shots=1024, seed=0, verbose=False,
                 class_weight=None, fake_backend="FakeBrisbane", opt_level=3):
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
        self.fake_backend = fake_backend
        self.opt_level = opt_level

    # ---- kernel -------------------------------------------------------------
    def _states(self, Z):
        return np.array([Statevector(self._fm_.assign_parameters(z)).data for z in Z])

    def _kernel(self, A, B=None):
        if self.kernel == "exact":
            SA = self._states(A)
            SB = SA if B is None else self._states(B)
            return np.abs(SA.conj() @ SB.T) ** 2
        if self.kernel in ("page", "page-noisy"):
            return self._overlap_kernel_.evaluate(A, B)
        raise ValueError(f"Unknown kernel '{self.kernel}'. Choose exact, page or page-noisy.")

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
        if self.kernel in ("page", "page-noisy"):
            self._overlap_kernel_ = OverlapKernel(
                self._fm_, mode=self.kernel, shots=self.shots, seed=self.seed,
                fake_backend=self.fake_backend, opt_level=self.opt_level, verbose=self.verbose,
            )
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
        if self.verbose and self.kernel != "exact":
            log("      test kernel ", end="")
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

    ap = argparse.ArgumentParser(description="Quantum kernel SVM for Golub leukemia data.")
    ap.add_argument("--data-dir", default=str(HERE.parent / "data"))
    ap.add_argument("--ks", "--k", dest="ks", type=int, nargs="+", default=[8],
                    help="Number of genes (= qubits) to select per fold; several values sweep")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0],
                    help="Multiplier applied after scaling (angle spread); several values sweep")
    ap.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    ap.add_argument("--feature-map", choices=["zz", "z"], default="zz")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--Cs", "--C", dest="Cs", type=float, nargs="+", default=[1.0],
                    help="SVC regularisation C; several values sweep")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--kernel", choices=["exact", "page", "page-noisy"], default="exact",
                    help="exact = ideal statevector overlaps; page = tutorial's overlap "
                         "circuits on StatevectorSampler; page-noisy = + transpile and "
                         "FakeBrisbane noise model")
    ap.add_argument("--shots", type=int, default=1024, help="Shots per overlap circuit")
    ap.add_argument("--fake-backend", default="FakeBrisbane",
                    help="Fake IBM device for page-noisy (name from qiskit_ibm_runtime.fake_provider)")
    ap.add_argument("--opt-level", type=int, default=3, help="Transpiler optimization level")
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

    kdesc = args.kernel + (f" ({args.shots} shots)" if args.kernel != "exact" else "")
    if args.kernel == "page-noisy":
        kdesc += f", {args.fake_backend} noise model, optimization_level={args.opt_level}"
    log(f"[2/4] Setup: {args.folds}-fold x {args.repeats} repeated CV; "
        f"{args.feature_map.upper()} map reps={args.reps}, scaler={args.scaler}, "
        f"class_weight={args.class_weight}, kernel={kdesc}")

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
            class_weight=class_weight, fake_backend=args.fake_backend,
            opt_level=args.opt_level,
        )
        r = evaluate(X, y, model, args.folds, args.repeats, args.seed)
        rows.append({
            "model": f"qsvc_{args.feature_map}_{args.kernel}", "k_genes": k,
            "scaler": args.scaler, "scale": scale, "reps": args.reps, "C": C,
            "class_weight": args.class_weight,
            "shots": 0 if args.kernel == "exact" else args.shots, **r,
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
        log(line)

    log("[4/4] Saving results ...")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"      Saved: {out}  (total time {time.perf_counter() - t_start:.1f}s)")


if __name__ == "__main__":
    _main()