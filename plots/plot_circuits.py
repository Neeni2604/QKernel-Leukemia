"""plot_circuits.py - Quantum circuit visualisation and kernel-matrix diagnostics.

Outputs (all dpi=200) to plots/circuit_images/:
  zz_feature_map.png          Parameterised ZZFeatureMap block view
  zz_feature_map_gates.png    Same circuit decomposed to elementary gates
  overlap_circuit.png         unitary_overlap(U(x_ALL), U(x_AML)) + measure_all
  shot_histogram.png          Shot histogram for the overlap circuit
  kernel_values.png           K(ALL,AML) with shot-noise interval (single pair)
  kernel_heatmap.png          Class-sorted 72x72 Gram matrix at scale=0.1 and 1.0

Also saves QML/results/kernel_class_stats.csv.

Usage:
    python plots/plot_circuits.py
    python plots/plot_circuits.py --k 8 --shots 2048 --out-dir plots/circuit_images
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import unitary_overlap, z_feature_map, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from data_loader import clip_and_log, load_golub
from selection import WelchTSelector

DPI = 200


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_patient_vectors(k: int, scale: float = 0.1):
    """Single ALL and AML patient vector for circuit demos."""
    d = load_golub()
    X = clip_and_log(d.X).to_numpy()
    y = d.y.to_numpy()
    sel = WelchTSelector(k=k).fit(X, y)
    X_sel = sel.transform(X)
    gene_labels = d.gene_info.index[sel.selected_idx_].tolist()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sel) * scale
    all_idx = int(np.where(y == 0)[0][0])
    aml_idx = int(np.where(y == 1)[0][0])
    return X_scaled[all_idx], X_scaled[aml_idx], gene_labels


def _load_all_patients(k: int, scale: float):
    """All 72 patients, sorted ALL-first. Returns (X_scaled, y_sorted, patient_ids)."""
    d = load_golub()
    X = clip_and_log(d.X).to_numpy()
    y = d.y.to_numpy()
    patient_ids = d.X.index.to_numpy()      # original patient numbers from the CSV
    sel = WelchTSelector(k=k).fit(X, y)
    X_sel = sel.transform(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sel) * scale
    order = np.argsort(y, kind="stable")    # ALL (y=0) first, AML (y=1) second
    return X_scaled[order], y[order], patient_ids[order]


# ── kernel helpers ─────────────────────────────────────────────────────────────

def _build_fm(name: str, k: int, reps: int = 1):
    if name == "zz":
        return zz_feature_map(feature_dimension=k, reps=reps, entanglement="linear")
    if name == "z":
        return z_feature_map(feature_dimension=k, reps=reps)
    raise ValueError(f"Unknown feature map '{name}'")


def compute_kernel_exact(X_scaled: np.ndarray, fm) -> np.ndarray:
    """Full kernel matrix via exact Statevector inner products (exported for reuse)."""
    n = len(X_scaled)
    svs = [Statevector(fm.assign_parameters(x)) for x in X_scaled]
    K = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = float(abs(svs[i].inner(svs[j])) ** 2)
            K[i, j] = K[j, i] = val
    return K


def pair_stats(K: np.ndarray, y_sorted: np.ndarray) -> dict:
    """Within-class (off-diagonal upper triangle) and between-class pair arrays."""
    idx_all = np.where(y_sorted == 0)[0]
    idx_aml = np.where(y_sorted == 1)[0]
    within_all = np.array([K[i, j] for i in idx_all for j in idx_all if j > i])
    within_aml = np.array([K[i, j] for i in idx_aml for j in idx_aml if j > i])
    between    = np.array([K[i, j] for i in idx_all for j in idx_aml])
    return dict(within_all=within_all, within_aml=within_aml, between=between,
                n_all=len(idx_all), n_aml=len(idx_aml),
                idx_all=idx_all, idx_aml=idx_aml)


def print_outliers(K: np.ndarray, y_sorted: np.ndarray,
                   patient_ids: np.ndarray, scale: float, stats: dict) -> None:
    """Print 5 lowest within-class outliers per class and any cross-class stragglers."""
    idx_all = stats["idx_all"]
    idx_aml = stats["idx_aml"]

    all_means = np.array([K[i, idx_all[idx_all != i]].mean() for i in idx_all])
    aml_means = np.array([K[i, idx_aml[idx_aml != i]].mean() for i in idx_aml])

    print(f"\n  [scale={scale}] 5 ALL patients with lowest mean within-ALL kernel:")
    for rank, pos in enumerate(np.argsort(all_means)[:5], 1):
        print(f"    #{rank} patient {patient_ids[idx_all[pos]]}: "
              f"mean={all_means[pos]:.4f}")

    print(f"  [scale={scale}] 5 AML patients with lowest mean within-AML kernel:")
    for rank, pos in enumerate(np.argsort(aml_means)[:5], 1):
        print(f"    #{rank} patient {patient_ids[idx_aml[pos]]}: "
              f"mean={aml_means[pos]:.4f}")

    print(f"  [scale={scale}] AML patients with higher mean K to ALL than to AML:")
    found = False
    for pos, i in enumerate(idx_aml):
        mean_to_all = K[i, idx_all].mean()
        mean_to_aml = K[i, idx_aml[idx_aml != i]].mean()
        if mean_to_all > mean_to_aml:
            print(f"    patient {patient_ids[i]}: "
                  f"K_to_ALL={mean_to_all:.4f}  K_to_AML={mean_to_aml:.4f}")
            found = True
    if not found:
        print("    (none)")


# ── shared drawing helper ──────────────────────────────────────────────────────

def draw_heatmap_row(fig, outer_gs, row_idx: int,
                     K: np.ndarray, y_sorted: np.ndarray,
                     title: str, stats: dict) -> None:
    """Draw one heatmap row: heatmap | colorbar axis | bar panel."""
    n_all = stats["n_all"]
    n_aml = stats["n_aml"]
    n_all_pairs = len(stats["within_all"])
    n_aml_pairs = len(stats["within_aml"])
    n_bet_pairs = len(stats["between"])

    inner = outer_gs[row_idx].subgridspec(1, 3,
                                          width_ratios=[20, 0.6, 7],
                                          wspace=0.30)
    ax_heat = fig.add_subplot(inner[0])
    ax_cax  = fig.add_subplot(inner[1])
    ax_bar  = fig.add_subplot(inner[2])

    # heatmap
    im = ax_heat.imshow(K, vmin=0, vmax=1, cmap="RdYlBu_r",
                        aspect="auto", interpolation="nearest")
    ax_heat.axhline(n_all - 0.5, color="white", linewidth=1.6, linestyle="--")
    ax_heat.axvline(n_all - 0.5, color="white", linewidth=1.6, linestyle="--")
    ax_heat.set_xticks([n_all / 2, n_all + n_aml / 2])
    ax_heat.set_xticklabels([f"ALL (n={n_all})", f"AML (n={n_aml})"], fontsize=10)
    ax_heat.set_yticks([n_all / 2, n_all + n_aml / 2])
    ax_heat.set_yticklabels([f"ALL\n(n={n_all})", f"AML\n(n={n_aml})"], fontsize=10)
    ax_heat.set_title(title, fontsize=11, fontweight="bold")

    # dedicated colorbar axis — no label overlap
    cb = fig.colorbar(im, cax=ax_cax)
    cb.set_label("K(xi, xj)", fontsize=9)

    # bar panel
    group_labels = [
        f"Within ALL\n({n_all} pts,\n{n_all_pairs} pairs)",
        f"Within AML\n({n_aml} pts,\n{n_aml_pairs} pairs)",
        f"ALL vs AML\n({n_bet_pairs} pairs)",
    ]
    group_means  = [stats["within_all"].mean(), stats["within_aml"].mean(),
                    stats["between"].mean()]
    group_stds   = [stats["within_all"].std(),  stats["within_aml"].std(),
                    stats["between"].std()]
    group_colors = ["#3AAB5E", "#8E5BB5", "#E06C3A"]

    x    = np.arange(3)
    bars = ax_bar.bar(x, group_means, yerr=group_stds, capsize=6, width=0.55,
                      color=group_colors, edgecolor="white", linewidth=0.7,
                      error_kw={"elinewidth": 1.5, "ecolor": "#333", "capthick": 1.5},
                      zorder=3)
    for bar, mean in zip(bars, group_means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    min(mean + 0.03, 1.08),
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(group_labels, fontsize=8)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel("Mean kernel value", fontsize=9)
    ax_bar.set_title("Mean ± std across pairs", fontsize=9, fontweight="bold")
    ax_bar.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax_bar.set_axisbelow(True)


# ── circuit / shot plots ───────────────────────────────────────────────────────

def plot_feature_map(k: int, out_dir: Path) -> None:
    fm = _build_fm("zz", k)

    fig = fm.draw("mpl", style="iqp", fold=-1)
    fig.suptitle(f"ZZFeatureMap  |  k={k} qubits, reps=1, entanglement=linear",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "zz_feature_map.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> zz_feature_map.png")

    fig = fm.decompose().draw("mpl", style="iqp", fold=40)
    fig.suptitle(f"ZZFeatureMap decomposed to elementary gates  |  k={k}",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "zz_feature_map_gates.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> zz_feature_map_gates.png")


def plot_overlap(x_all: np.ndarray, x_aml: np.ndarray, out_dir: Path) -> None:
    k  = len(x_all)
    fm = _build_fm("zz", k)
    overlap = unitary_overlap(fm.assign_parameters(x_all), fm.assign_parameters(x_aml))
    overlap.measure_all()

    fig = overlap.draw("mpl", style="iqp", fold=-1)
    fig.suptitle(
        f"Overlap circuit  U(x_ALL)†  U(x_AML)  |  k={k}\n"
        "P(all zeros) = K(ALL, AML) = |<0| U(ALL)† U(AML) |0>|^2",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.savefig(out_dir / "overlap_circuit.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> overlap_circuit.png")


def plot_shots(x_all: np.ndarray, x_aml: np.ndarray,
               shots: int, out_dir: Path) -> None:
    k         = len(x_all)
    fm        = _build_fm("zz", k)
    zeros_key = "0" * k

    def _run(xa, xb):
        circ = unitary_overlap(fm.assign_parameters(xa), fm.assign_parameters(xb))
        circ.measure_all()
        result = StatevectorSampler(default_shots=shots, seed=42).run([(circ,)]).result()
        counts = result[0].data.meas.get_counts()
        return counts, counts.get(zeros_key, 0) / shots

    print(f"  running overlap circuits ({shots} shots each)...")
    counts_am, k_am = _run(x_all, x_aml)

    # shot histogram
    top20      = dict(sorted(counts_am.items(), key=lambda kv: -kv[1])[:20])
    bitstrings = list(top20.keys())
    colors     = ["#F0C040" if b == zeros_key else "#2F7EBF" for b in bitstrings]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(bitstrings)), [top20[b] for b in bitstrings],
           color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(bitstrings)))
    ax.set_xticklabels(bitstrings, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Measurement outcome (bitstring)", fontsize=10)
    ax.set_ylabel("Counts", fontsize=10)
    ax.set_title(
        f"Shot histogram  |  overlap circuit U(x_ALL)† U(x_AML)  |  {shots} shots  |  k={k}\n"
        f"Gold bar = all-zeros outcome  ->  K(ALL, AML) = {k_am:.4f}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(handles=[
        mpatches.Patch(color="#F0C040", label=f"|{'0'*k}> (kernel value = {k_am:.4f})"),
        mpatches.Patch(color="#2F7EBF", label="Other outcomes"),
    ], fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "shot_histogram.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> shot_histogram.png")

    # single-pair kernel value bar with shot-noise interval
    # shot noise: +/- sqrt(p*(1-p)/shots)
    noise = float(np.sqrt(k_am * (1 - k_am) / shots))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bar = ax.bar([0], [k_am], yerr=[[noise], [noise]],
                 capsize=10, width=0.4, color="#E06C3A",
                 edgecolor="white", linewidth=0.8, zorder=3,
                 error_kw={"elinewidth": 2, "ecolor": "#333", "capthick": 2})
    ax.text(0, k_am + noise + 0.025, f"{k_am:.4f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["K(ALL, AML)"], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Kernel value", fontsize=11)
    ax.set_title(
        f"One example kernel entry: ALL vs AML patient ({shots} shots)\n"
        f"k={k} genes, ZZFeatureMap reps=1, scale=0.1",
        fontsize=10, fontweight="bold",
    )
    fig.text(0.5, 0.01,
             "Illustrates the overlap circuit; not evidence of class separation.\n"
             f"Error bar = shot-noise interval +/- sqrt(p(1-p)/shots) = +/-{noise:.4f}",
             ha="center", va="bottom", fontsize=7.5, color="#555555")
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_dir / "kernel_values.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> kernel_values.png")


# ── heatmap + stats ────────────────────────────────────────────────────────────

def plot_kernel_heatmap(k: int, out_dir: Path) -> None:
    """Class-sorted Gram matrix at scale=0.1 and 1.0, with within/between stats."""
    scales      = [0.1, 1.0]
    csv_rows    = []
    fig         = plt.figure(figsize=(18, 13))
    outer       = fig.add_gridspec(2, 1, hspace=0.45)

    for row_idx, scale in enumerate(scales):
        print(f"  computing kernel matrix at scale={scale} ...")
        X_scaled, y_sorted, patient_ids = _load_all_patients(k=k, scale=scale)
        fm = _build_fm("zz", k)
        K  = compute_kernel_exact(X_scaled, fm)
        st = pair_stats(K, y_sorted)

        draw_heatmap_row(fig, outer, row_idx, K, y_sorted,
                         title=f"ZZFeatureMap kernel matrix  |  scale={scale}",
                         stats=st)
        print_outliers(K, y_sorted, patient_ids, scale, st)

        # collect CSV rows
        for group, arr in [("within_all", st["within_all"]),
                            ("within_aml", st["within_aml"]),
                            ("between",    st["between"])]:
            csv_rows.append(dict(kernel="zz", scale=scale, group=group,
                                 mean=round(float(arr.mean()), 6),
                                 std=round(float(arr.std()), 6),
                                 n_pairs=len(arr)))

        # print table row
        print(f"  scale={scale}  within-ALL: {st['within_all'].mean():.4f} "
              f"+/-{st['within_all'].std():.4f} (n={len(st['within_all'])})  "
              f"within-AML: {st['within_aml'].mean():.4f} "
              f"+/-{st['within_aml'].std():.4f} (n={len(st['within_aml'])})  "
              f"between: {st['between'].mean():.4f} "
              f"+/-{st['between'].std():.4f} (n={len(st['between'])})")

    fig.suptitle(
        f"Quantum kernel (ZZFeatureMap, k={k} genes, reps=1)  |  patients sorted ALL first\n"
        "White dashed line = class boundary\n"
        "NOTE: gene selection and scaling fit on all 72 patients; "
        "block structure is optimistic. CV results are the performance claim.",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.savefig(out_dir / "kernel_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> kernel_heatmap.png")

    # save CSV
    csv_path = ROOT / "QML" / "results" / "kernel_class_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  saved -> {csv_path.relative_to(ROOT)}")

    # print table
    df = pd.DataFrame(csv_rows)
    print("\nKernel class stats table:")
    print(df.to_string(index=False))


# ── main ───────────────────────────────────────────────────────────────────────

def main(k: int, shots: int, scale: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading patient data...")
    x_all, x_aml, gene_labels = _load_patient_vectors(k=k, scale=scale)
    print(f"  k={k} genes, scale={scale}")
    print(f"  x_ALL (first 4): {x_all[:4].round(4)}")
    print(f"  x_AML (first 4): {x_aml[:4].round(4)}")

    print("\nDrawing feature map circuits...")
    plot_feature_map(k, out_dir)

    print("\nDrawing overlap circuit...")
    plot_overlap(x_all, x_aml, out_dir)

    print("\nRunning shot simulation...")
    plot_shots(x_all, x_aml, shots, out_dir)

    print("\nComputing kernel matrices (scale=0.1 and scale=1.0)...")
    plot_kernel_heatmap(k, out_dir)

    print(f"\nAll circuit plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k",       type=int,   default=8)
    ap.add_argument("--shots",   type=int,   default=2048)
    ap.add_argument("--scale",   type=float, default=0.1)
    ap.add_argument("--out-dir", default=str(HERE / "circuit_images"))
    args = ap.parse_args()
    main(k=args.k, shots=args.shots, scale=args.scale, out_dir=Path(args.out_dir))
