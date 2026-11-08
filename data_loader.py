"""data_loader.py - load the Golub leukemia dataset (ALL vs AML) for QKernel Leukemia.

The original Golub files are laid out with genes as ROWS and patients as COLUMNS,
with a "call" column (Affymetrix P/A/M flag) after every patient. This module
turns them into the usual machine-learning shape: patients x genes.

Expected files in the data directory (default: ./data):
    data_set_ALL_AML_train.csv         patients 1-38
    data_set_ALL_AML_independent.csv   patients 39-72
    actual.csv                         columns: patient, cancer (ALL / AML)

Usage:
    python data_loader.py                      # load + print sanity checks
    python data_loader.py --data-dir path/to/data

    from data_loader import load_golub, clip_and_log
    d = load_golub("data")
    X, y = d.X, d.y                     # raw intensities, 0 = ALL, 1 = AML
    X_log = clip_and_log(X)             # fixed transform, safe to apply before CV

Nothing here learns from the data, so it is safe to run before cross-validation.
Anything that fits to data (scaling, gene selection, models) belongs inside the folds.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FILE = "data_set_ALL_AML_train.csv"
TEST_FILE = "data_set_ALL_AML_independent.csv"
LABEL_FILE = "actual.csv"

DESC_COL = "Gene Description"
ACC_COL = "Gene Accession Number"

EXPECTED_SAMPLES = 72
EXPECTED_CLASS_COUNTS = {"ALL": 47, "AML": 25}
EXPECTED_GENES_RANGE = (7000, 7200)  # about 7129 in the standard files

_DEFAULT_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class GolubData:
    X: pd.DataFrame          # patients x genes, raw intensities (float)
    y: pd.Series             # 0 = ALL, 1 = AML, indexed by patient number
    is_train: pd.Series      # True for the 38 original Golub training patients
    gene_info: pd.DataFrame  # index = gene_id; columns = accession, description


def _read_expression(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one expression CSV -> (gene info, genes x patients values)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # pandas renames repeated 'call' headers to call.1, call.2, ...
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("call")])

    missing = {DESC_COL, ACC_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected column(s) {sorted(missing)}")

    info = df[[ACC_COL, DESC_COL]].rename(
        columns={ACC_COL: "accession", DESC_COL: "description"}
    )
    values = df.drop(columns=[ACC_COL, DESC_COL])
    try:
        values.columns = values.columns.astype(int)
    except ValueError as e:
        raise ValueError(
            f"{path.name}: non-numeric patient column after dropping 'call' columns"
        ) from e
    values = values.apply(pd.to_numeric, errors="coerce")
    return info.reset_index(drop=True), values.reset_index(drop=True)


def _make_gene_ids(accessions: pd.Series) -> pd.Series:
    """Accession numbers can repeat; add _1, _2, ... to later repeats."""
    n = accessions.groupby(accessions).cumcount()
    return accessions.where(n == 0, accessions + "_" + n.astype(str))


def _read_labels(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    lab = pd.read_csv(path)
    lab.columns = [str(c).strip().lower() for c in lab.columns]
    if not {"patient", "cancer"} <= set(lab.columns):
        raise ValueError(
            f"{path.name}: expected columns 'patient' and 'cancer', found {list(lab.columns)}"
        )
    s = lab.set_index("patient")["cancer"].astype(str).str.strip().str.upper()
    s.index = s.index.astype(int)
    return s


def load_golub(data_dir: str | Path = _DEFAULT_DATA_DIR) -> GolubData:
    """Load and pool all 72 Golub patients. Raw intensities, no transforms."""
    data_dir = Path(data_dir)
    tr_info, tr_vals = _read_expression(data_dir / TRAIN_FILE)
    te_info, te_vals = _read_expression(data_dir / TEST_FILE)

    if len(tr_info) != len(te_info) or not (
        tr_info["accession"].values == te_info["accession"].values
    ).all():
        raise ValueError("Train and test files do not list the same genes in the same order.")

    gene_ids = _make_gene_ids(tr_info["accession"])
    gene_info = tr_info.copy()
    gene_info.index = pd.Index(gene_ids, name="gene_id")

    def to_patients_by_genes(vals: pd.DataFrame) -> pd.DataFrame:
        out = vals.T
        out.columns = gene_info.index
        return out

    X_tr, X_te = to_patients_by_genes(tr_vals), to_patients_by_genes(te_vals)
    X = pd.concat([X_tr, X_te]).sort_index().astype(float)
    X.index.name = "patient"

    if X.index.duplicated().any():
        raise ValueError("Duplicate patient numbers across train and test files.")

    labels = _read_labels(data_dir / LABEL_FILE)
    absent = X.index.difference(labels.index)
    if len(absent):
        raise ValueError(f"No label for patients: {list(absent)}")
    unknown = set(labels.loc[X.index]) - set(EXPECTED_CLASS_COUNTS)
    if unknown:
        raise ValueError(f"Unexpected label values: {unknown}")

    y = (labels.loc[X.index] == "AML").astype(int)
    y.name = "AML"
    is_train = pd.Series(X.index.isin(X_tr.index), index=X.index, name="is_train")

    return GolubData(X=X, y=y, is_train=is_train, gene_info=gene_info)


def clip_and_log(X: pd.DataFrame, floor: float = 100.0, ceil: float = 16000.0) -> pd.DataFrame:
    """Clip raw intensities to [floor, ceil], then log2.

    Floor/ceiling follow the thresholds commonly used with the original Golub
    preprocessing; change them if you want. It is a fixed transform (learns
    nothing from the data), so it is safe to apply before cross-validation.
    """
    return np.log2(X.clip(lower=floor, upper=ceil))


def validate(d: GolubData, strict: bool = True) -> list[str]:
    """Return a list of problems; raise on hard failures if strict."""
    problems: list[str] = []
    n, g = d.X.shape

    if n != EXPECTED_SAMPLES:
        problems.append(f"expected {EXPECTED_SAMPLES} patients, got {n}")
    lo, hi = EXPECTED_GENES_RANGE
    if not lo <= g <= hi:
        problems.append(f"expected roughly 7129 genes, got {g}")

    counts = d.y.map({0: "ALL", 1: "AML"}).value_counts().to_dict()
    if counts != EXPECTED_CLASS_COUNTS:
        problems.append(f"expected class counts {EXPECTED_CLASS_COUNTS}, got {counts}")

    nan = int(d.X.isna().sum().sum())
    if nan:
        problems.append(f"{nan} missing / non-numeric values in X")

    hard = [p for p in problems if "patients" in p or "class counts" in p or "missing" in p]
    if strict and hard:
        raise ValueError("Data validation failed: " + "; ".join(hard))
    for p in problems:
        if p not in hard:
            warnings.warn(p)
    return problems


def _main() -> None:
    ap = argparse.ArgumentParser(description="Load Golub data and run sanity checks.")
    ap.add_argument("--data-dir", default=str(_DEFAULT_DATA_DIR))
    args = ap.parse_args()

    d = load_golub(args.data_dir)
    problems = validate(d, strict=False)

    X, y = d.X, d.y
    print(f"X shape (patients x genes): {X.shape}")
    print("Class counts:", y.map({0: "ALL", 1: "AML"}).value_counts().to_dict())
    print(
        "Original split: "
        f"{int(d.is_train.sum())} train / {int((~d.is_train).sum())} test patients"
    )
    print(f"Raw value range: {np.nanmin(X.values):.1f} to {np.nanmax(X.values):.1f}")
    print(f"Majority-class baseline accuracy: {max(y.mean(), 1 - y.mean()):.3f}")
    print(f"Unique gene ids: {X.columns.is_unique} ({X.shape[1] - d.gene_info['accession'].nunique()} repeated accessions)")
    print("Checks:", "all good" if not problems else problems)


if __name__ == "__main__":
    _main()
