#!/usr/bin/env python3
"""Merge the ALL/AML training and independent gene-expression CSV files.

Rows are matched by Gene Accession Number. Samples (expression + detection
call) are ordered numerically. Duplicate call headers are renamed <ID>_call.

Usage:
    python combine_all_aml.py "data_set_ALL_AML_train(1).csv" \
        data_set_ALL_AML_independent.csv -o ALL_AML_combined.csv
"""

import argparse
import csv
from pathlib import Path

GENE_COLUMNS = ("Gene Description", "Gene Accession Number")


def load_data(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        if tuple(header[:2]) != GENE_COLUMNS or (len(header) - 2) % 2:
            raise ValueError(f"Unexpected CSV structure: {path}")

        # Each sample is represented by an expression-value column and a call column.
        samples = {}
        for col in range(2, len(header), 2):
            sample = header[col].strip()
            if not sample.isdigit() or header[col + 1].strip() != "call":
                raise ValueError(f"Unexpected sample columns in {path}: {header[col:col+2]}")
            if sample in samples:
                raise ValueError(f"Duplicate sample {sample} in {path}")
            samples[sample] = col

        genes = {}
        for row in reader:
            if not row:  # Ignore blank lines present in the source CSVs.
                continue
            if len(row) != len(header):
                raise ValueError(f"Incorrect column count in {path} for gene {row[:2]}")
            accession = row[1]
            if not accession or accession in genes:
                raise ValueError(f"Missing or repeated gene accession {accession!r} in {path}")
            genes[accession] = row

    return genes, samples


def combine(train_path, independent_path, output_path):
    train, train_samples = load_data(train_path)
    independent, independent_samples = load_data(independent_path)

    if set(train) != set(independent):
        raise ValueError("The files have different gene accession sets; cannot safely merge.")
    if set(train_samples) & set(independent_samples):
        raise ValueError("The files contain overlapping sample IDs.")

    # Store the originating dataset and the expression column for every sample.
    samples = [(int(s), train, col) for s, col in train_samples.items()]
    samples += [(int(s), independent, col) for s, col in independent_samples.items()]
    samples.sort(key=lambda item: item[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = list(GENE_COLUMNS)
        for sample_id, _, _ in samples:
            header.extend([str(sample_id), f"{sample_id}_call"])
        writer.writerow(header)

        for accession, train_row in train.items():
            independent_row = independent[accession]
            if train_row[0] != independent_row[0]:
                raise ValueError(f"Gene description differs for {accession}")
            output_row = train_row[:2]
            for _, source, col in samples:
                output_row.extend(source[accession][col:col + 2])
            writer.writerow(output_row)

    print(f"Saved {len(train):,} genes and {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path, help="Training CSV file")
    parser.add_argument("independent", type=Path, help="Independent CSV file")
    parser.add_argument("-o", "--output", type=Path, default=Path("ALL_AML_combined.csv"))
    args = parser.parse_args()
    combine(args.train, args.independent, args.output)
