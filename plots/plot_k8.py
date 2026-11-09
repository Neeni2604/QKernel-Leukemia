"""plot_k8.py - Bar charts for every model at k=8 genes.

Reads:
    baselines/results/k_sweep.csv       classical models
    QML/results/final_default.csv       QSVC default  (scale=1.0)
    QML/results/final_tuned.csv         QSVC tuned    (scale=0.1)

Writes one PNG per model into plots/images/.

Usage:
    python plots/plot_k8.py
    python plots/plot_k8.py --baselines baselines/results/k_sweep.csv
                             --qsvc-default QML/results/final_default.csv
                             --qsvc-tuned   QML/results/final_tuned.csv
                             --out-dir      plots/images
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── consistent metric colours ────────────────────────────────────────────────
METRIC_COLORS = {
    "accuracy":    "#2F7EBF",   # steel blue
    "auroc":       "#E06C3A",   # burnt orange
    "sensitivity": "#3AAB5E",   # green
    "specificity": "#8E5BB5",   # purple
}
METRICS = list(METRIC_COLORS)
METRIC_LABELS = {
    "accuracy":    "Accuracy",
    "auroc":       "AUROC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}

MAJORITY_ACC = 0.6524   # majority-class baseline from the data

MODEL_DISPLAY = {
    "majority_class":        "Majority Class",
    "logistic_regression":   "Logistic Regression",
    "svm_linear":            "SVM (Linear)",
    "svm_rbf":               "SVM (RBF)",
    "random_forest":         "Random Forest",
    "qsvc_zz_exact_default": "QSVC ZZ-Exact\n(default scale=1.0)",
    "qsvc_zz_exact_tuned":   "QSVC ZZ-Exact\n(tuned  scale=0.1)",
}


def _plot_model(row: pd.Series, title: str, out_path: Path, is_quantum: bool = False) -> None:
    means = [row[f"{m}_mean"] for m in METRICS]
    stds  = [row[f"{m}_std"]  for m in METRICS]
    colors = [METRIC_COLORS[m] for m in METRICS]
    labels = [METRIC_LABELS[m] for m in METRICS]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(METRICS))
    bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.55,
                  color=colors, edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.5, "ecolor": "#333333", "capthick": 1.5},
                  zorder=3)

    # value labels above each bar
    for bar, mean, std in zip(bars, means, stds):
        ypos = min(mean + std + 0.012, 0.98)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, color="#222222")

    # majority-class baseline
    ax.axhline(MAJORITY_ACC, color="#888888", linewidth=1.3,
               linestyle="--", zorder=2, label=f"Guess ALL ({MAJORITY_ACC:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    subtitle = "Quantum kernel (ZZFeatureMap, exact)" if is_quantum else "Classical baseline"
    ax.set_title(f"{title}\n"
                 f"k = 8 genes · {subtitle}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="lower right")

    # error-bar legend note
    fig.text(0.98, 0.01, "Error bars = ±1 std (fold-to-fold)",
             ha="right", va="bottom", fontsize=7.5, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path.name}")


def _plot_comparison(rows: list[tuple[str, pd.Series, bool]], out_path: Path) -> None:
    """Grouped bar chart: one group per metric, one bar per model."""
    model_names  = [name for name, _, _ in rows]
    is_quantum   = [q    for _, _, q   in rows]

    # bar colours: quantum models get a hatched gold, classical get their model colour
    MODEL_BAR_COLORS = [
        "#555555",  # majority class  - dark grey
        "#2F7EBF",  # logistic        - blue
        "#5BA85B",  # svm linear      - green
        "#E06C3A",  # svm rbf         - orange
        "#8E5BB5",  # random forest   - purple
        "#F0C040",  # qsvc default    - gold
        "#F0C040",  # qsvc tuned      - gold (hatched differently)
    ]
    hatches = ["", "", "", "", "", "//", "xx"]

    n_models  = len(rows)
    n_metrics = len(METRICS)
    group_w   = 0.8
    bar_w     = group_w / n_models
    group_x   = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(13, 6))

    for m_idx, (name, row, _) in enumerate(rows):
        offsets = group_x + (m_idx - n_models / 2 + 0.5) * bar_w
        means   = [row[f"{m}_mean"] for m in METRICS]
        stds    = [row[f"{m}_std"]  for m in METRICS]
        color   = MODEL_BAR_COLORS[m_idx]
        hatch   = hatches[m_idx]
        ax.bar(offsets, means, width=bar_w * 0.92,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.6,
               yerr=stds, capsize=3,
               error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capthick": 1.2},
               label=name, zorder=3)

    ax.axhline(MAJORITY_ACC, color="#888888", linewidth=1.3,
               linestyle="--", zorder=2, label=f"Guess ALL ({MAJORITY_ACC:.3f})")

    ax.set_xticks(group_x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=12)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("All Models Comparison  |  k = 8 genes",
                 fontsize=13, fontweight="bold", pad=12)

    ax.legend(fontsize=8.5, loc="lower right", ncol=2,
              framealpha=0.9, edgecolor="#cccccc")
    fig.text(0.98, 0.01, "Error bars = ±1 std (fold-to-fold)  |  Gold bars = QSVC quantum kernel",
             ha="right", va="bottom", fontsize=7.5, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path.name}")


def main(baselines_csv: Path, qsvc_default_csv: Path, qsvc_tuned_csv: Path,
         out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── classical models at k=8 ──────────────────────────────────────────────
    df = pd.read_csv(baselines_csv)
    df8 = df[df["k_genes"] == 8].copy()
    if df8.empty:
        raise ValueError(f"No rows with k_genes==8 in {baselines_csv}")

    print("Classical models:")
    classical_rows: list[tuple[str, pd.Series, bool]] = []
    for _, row in df8.iterrows():
        model_key = row["model"]
        display   = MODEL_DISPLAY.get(model_key, model_key.replace("_", " ").title())
        fname     = f"classical_{model_key}_k8.png"
        _plot_model(row, display, out_dir / fname, is_quantum=False)
        classical_rows.append((display, row, False))

    # ── QSVC models ──────────────────────────────────────────────────────────
    print("QSVC models:")
    qsvc_rows: list[tuple[str, pd.Series, bool]] = []
    for label, csv_path, key in [
        ("default", qsvc_default_csv, "qsvc_zz_exact_default"),
        ("tuned",   qsvc_tuned_csv,   "qsvc_zz_exact_tuned"),
    ]:
        if not csv_path.exists():
            print(f"  skipping {label} - file not found: {csv_path}")
            continue
        row = pd.read_csv(csv_path).iloc[0]
        display = MODEL_DISPLAY[key]
        fname   = f"qsvc_{label}_k8.png"
        _plot_model(row, display, out_dir / fname, is_quantum=True)
        qsvc_rows.append((display.replace("\n", " "), row, True))

    # ── comparison plot ───────────────────────────────────────────────────────
    print("Comparison plot:")
    all_rows = classical_rows + qsvc_rows
    _plot_comparison(all_rows, out_dir / "comparison_all_models_k8.png")

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent   # project root

    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines",     default=str(root / "baselines" / "results" / "k_sweep.csv"))
    ap.add_argument("--qsvc-default",  default=str(root / "QML" / "results" / "final_default.csv"))
    ap.add_argument("--qsvc-tuned",    default=str(root / "QML" / "results" / "final_tuned.csv"))
    ap.add_argument("--out-dir",       default=str(Path(__file__).resolve().parent / "images"))
    args = ap.parse_args()

    main(
        baselines_csv    = Path(args.baselines),
        qsvc_default_csv = Path(args.qsvc_default),
        qsvc_tuned_csv   = Path(args.qsvc_tuned),
        out_dir          = Path(args.out_dir),
    )
