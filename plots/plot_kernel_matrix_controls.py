"""plot_kernel_matrix_controls.py - Control comparison for the quantum kernel.

Produces plots/kernel_matrix_controls.png (dpi=200):
  Row 0: ZZ quantum kernel, scale=0.1
  Row 1: Z  quantum kernel, scale=0.1  (classically simulable, no entanglement)
  Row 2: Classical RBF kernel, scale=0.1, gamma matched to ZZ median off-diagonal

For each kernel also reports:
  - within-minus-between gap
  - 5x5 repeated stratified CV accuracy and AUROC (C=1)

Prints a one-line summary without claiming quantum advantage.

Usage:
    python plots/plot_kernel_matrix_controls.py
    python plots/plot_kernel_matrix_controls.py --folds 3 --repeats 1   # faster
"""
from __future__ import annotations

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))          # for plot_circuits helpers
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "QML"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from plot_circuits import (
    _build_fm,
    _load_all_patients,
    compute_kernel_exact,
    draw_heatmap_row,
    pair_stats,
)
from data_loader import clip_and_log, load_golub
from selection import WelchTSelector
from qsvc_pipeline import QuantumKernelPipeline

DPI = 200

SCORING = {
    "accuracy": "accuracy",
    "auroc":    "roc_auc",
    "sensitivity": make_scorer(recall_score, pos_label=1),
    "specificity": make_scorer(recall_score, pos_label=0),
}


# ── RBF fold-safe pipeline ────────────────────────────────────────────────────

class RBFKernelPipeline(BaseEstimator, ClassifierMixin):
    """WelchT selection -> MinMaxScaler*scale -> RBF(precomputed) -> SVC."""

    def __init__(self, k: int = 8, scale: float = 0.1,
                 C: float = 1.0, gamma: float = 1.0):
        self.k = k
        self.scale = scale
        self.C = C
        self.gamma = gamma

    def _preprocess(self, X, y=None, fit: bool = False):
        if fit:
            self.selector_ = WelchTSelector(k=self.k).fit(X, y)
        X_sel = self.selector_.transform(X)
        if fit:
            self.scaler_ = MinMaxScaler().fit(X_sel)
        return self.scaler_.transform(X_sel) * self.scale

    def fit(self, X, y):
        X_tr = self._preprocess(X, y, fit=True)
        K = sklearn_rbf(X_tr, X_tr, gamma=self.gamma)
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True)
        self.svc_.fit(K, y)
        self.X_train_ = X_tr
        return self

    def predict(self, X):
        return self.svc_.predict(
            sklearn_rbf(self._preprocess(X), self.X_train_, gamma=self.gamma))

    def predict_proba(self, X):
        return self.svc_.predict_proba(
            sklearn_rbf(self._preprocess(X), self.X_train_, gamma=self.gamma))

    def decision_function(self, X):
        return self.svc_.decision_function(
            sklearn_rbf(self._preprocess(X), self.X_train_, gamma=self.gamma))


# ── gamma matching ─────────────────────────────────────────────────────────────

def _match_gamma(X_scaled: np.ndarray, K_zz: np.ndarray) -> float:
    """gamma s.t. exp(-gamma * median_sqdist) == median off-diagonal K_zz."""
    n = len(X_scaled)
    off_K    = np.array([K_zz[i, j] for i in range(n) for j in range(i + 1, n)])
    sq_dists = np.array([np.sum((X_scaled[i] - X_scaled[j]) ** 2)
                         for i in range(n) for j in range(i + 1, n)])
    med_K  = float(np.median(off_K))
    med_sq = float(np.median(sq_dists))
    if med_K <= 1e-12 or med_sq <= 1e-12:
        return 1.0
    return float(-np.log(med_K) / med_sq)


# ── CV helper ─────────────────────────────────────────────────────────────────

def _run_cv(estimator, X, y, n_splits, n_repeats, seed=0) -> dict:
    cv  = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=seed)
    res = cross_validate(estimator, X, y, cv=cv,
                         scoring=SCORING, error_score="raise")
    return {m: (float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std()))
            for m in SCORING}


# ── main plot ──────────────────────────────────────────────────────────────────

def main(k: int = 8, scale: float = 0.1,
         folds: int = 5, repeats: int = 5,
         out_path: Path = HERE / "kernel_matrix_controls.png") -> None:

    print("Loading patient data ...")
    X_scaled, y_sorted, patient_ids = _load_all_patients(k=k, scale=scale)
    d  = load_golub()
    X_raw = clip_and_log(d.X).to_numpy()
    y_raw = d.y.to_numpy()

    # ── compute all three kernel matrices ─────────────────────────────────────
    print("Computing ZZ kernel matrix ...")
    fm_zz = _build_fm("zz", k)
    K_zz  = compute_kernel_exact(X_scaled, fm_zz)

    print("Computing Z kernel matrix ...")
    fm_z = _build_fm("z", k)
    K_z  = compute_kernel_exact(X_scaled, fm_z)

    gamma = _match_gamma(X_scaled, K_zz)
    print(f"Computing RBF kernel matrix (matched gamma={gamma:.4f}) ...")
    K_rbf = sklearn_rbf(X_scaled, X_scaled, gamma=gamma)

    kernels = [
        ("ZZ quantum (scale=0.1)",       K_zz,  "#2F7EBF"),
        ("Z quantum / no entanglement",  K_z,   "#3AAB5E"),
        (f"Classical RBF (gamma={gamma:.3f})", K_rbf, "#E06C3A"),
    ]

    # ── CV ────────────────────────────────────────────────────────────────────
    print(f"\nRunning {folds}x{repeats} CV for all kernels "
          f"(C=1, k={k}, scale={scale}) ...")
    cv_results = {}

    print("  CV: ZZ quantum ...")
    cv_results["ZZ"] = _run_cv(
        QuantumKernelPipeline(k=k, feature_map="zz", scale=scale,
                              kernel="exact", C=1.0),
        X_raw, y_raw, folds, repeats)

    print("  CV: Z quantum ...")
    cv_results["Z"] = _run_cv(
        QuantumKernelPipeline(k=k, feature_map="z", scale=scale,
                              kernel="exact", C=1.0),
        X_raw, y_raw, folds, repeats)

    print("  CV: RBF ...")
    cv_results["RBF"] = _run_cv(
        RBFKernelPipeline(k=k, scale=scale, C=1.0, gamma=gamma),
        X_raw, y_raw, folds, repeats)

    # ── print stats table ─────────────────────────────────────────────────────
    print("\n--- Control kernel stats ---")
    print(f"{'Kernel':<35} {'gap':>7}  {'acc':>13}  {'auroc':>13}")
    summaries = []
    for label, K, _ in kernels:
        st = pair_stats(K, y_sorted)
        gap = float(st["within_all"].mean() + st["within_aml"].mean()) / 2 \
              - float(st["between"].mean())
        cv_key = "ZZ" if "ZZ" in label else ("Z" if "Z quantum" in label else "RBF")
        acc_m,  acc_s  = cv_results[cv_key]["accuracy"]
        auroc_m, auroc_s = cv_results[cv_key]["auroc"]
        print(f"  {label:<33} {gap:+.4f}  "
              f"{acc_m:.3f}+/-{acc_s:.3f}  "
              f"{auroc_m:.3f}+/-{auroc_s:.3f}")
        summaries.append((label, gap, acc_m, auroc_m))

    # one-line summary (no quantum advantage claim)
    gaps  = [s[1] for s in summaries]
    accs  = [s[2] for s in summaries]
    names = [s[0] for s in summaries]
    best_gap_name = names[int(np.argmax(gaps))]
    best_acc_name = names[int(np.argmax(accs))]
    zz_gap, z_gap, rbf_gap = gaps
    zz_acc, z_acc, rbf_acc = accs
    if abs(zz_acc - rbf_acc) < 0.02:
        summary = (f"ZZ and RBF have similar CV accuracy ({zz_acc:.3f} vs {rbf_acc:.3f}); "
                   f"ZZ gap={zz_gap:+.4f}, RBF gap={rbf_gap:+.4f}. "
                   "Cannot attribute any difference to quantum effects at this scale.")
    elif zz_acc > rbf_acc + 0.02:
        summary = (f"ZZ outperforms RBF ({zz_acc:.3f} vs {rbf_acc:.3f}) at this encoding; "
                   f"further analysis needed to rule out bandwidth artefacts.")
    else:
        summary = (f"RBF matches or beats ZZ ({rbf_acc:.3f} vs {zz_acc:.3f}); "
                   f"ZZ at scale={scale} behaves like a smooth classical kernel.")
    print(f"\nSummary: {summary}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig    = plt.figure(figsize=(18, 18))
    outer  = fig.add_gridspec(3, 1, hspace=0.52)

    for row_idx, (label, K, _) in enumerate(kernels):
        st = pair_stats(K, y_sorted)
        cv_key = "ZZ" if "ZZ" in label else ("Z" if "Z quantum" in label else "RBF")
        acc_m, acc_s   = cv_results[cv_key]["accuracy"]
        auroc_m, auroc_s = cv_results[cv_key]["auroc"]
        gap = float(st["within_all"].mean() + st["within_aml"].mean()) / 2 \
              - float(st["between"].mean())

        title = (f"{label}\n"
                 f"gap={gap:+.4f}   CV acc={acc_m:.3f}+/-{acc_s:.3f}   "
                 f"AUROC={auroc_m:.3f}+/-{auroc_s:.3f}")
        draw_heatmap_row(fig, outer, row_idx, K, y_sorted, title=title, stats=st)

    fig.suptitle(
        f"Kernel comparison  |  k={k} genes, scale={scale}, C=1\n"
        "Gene selection and scaling fit on all 72 patients (optimistic for visualization).\n"
        f"CV: {folds}x{repeats} repeated stratified k-fold. "
        "RBF gamma matched to ZZ median off-diagonal.",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.text(0.5, -0.01, summary, ha="center", va="top",
             fontsize=9, color="#444444",
             wrap=True, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k",       type=int,   default=8)
    ap.add_argument("--scale",   type=float, default=0.1)
    ap.add_argument("--folds",   type=int,   default=5)
    ap.add_argument("--repeats", type=int,   default=5)
    ap.add_argument("--out",     default=str(HERE / "kernel_matrix_controls.png"))
    args = ap.parse_args()
    main(k=args.k, scale=args.scale, folds=args.folds, repeats=args.repeats,
         out_path=Path(args.out))
