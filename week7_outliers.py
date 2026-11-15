# Week 7: Flag IQR outliers and create separate filtered datasets.
# Run from the IDX Exchange folder: python week7_outliers.py

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output")
METRICS = ["ClosePrice", "LivingArea", "DaysOnMarket"]
CHUNK_SIZE = 10_000


def valid_for_iqr(values, column):
    """Use plausible numeric values to calculate the IQR boundaries."""
    values = pd.to_numeric(values, errors="coerce")
    valid = np.isfinite(values)

    if column == "DaysOnMarket":
        valid &= values >= 0
    else:
        valid &= values > 0

    return values[valid]


def get_boundaries(source):
    # Read just three columns to calculate boundaries without loading
    # the entire wide property dataset into memory.
    columns = pd.read_csv(source, usecols=METRICS, low_memory=False)
    boundaries = {}

    for column in METRICS:
        valid = valid_for_iqr(columns[column], column)

        if valid.empty:
            boundaries[column] = None
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        boundaries[column] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": q1 - 1.5 * iqr,
            "upper": q3 + 1.5 * iqr,
            "p01": valid.quantile(0.01),
            "p99": valid.quantile(0.99),
        }

    return boundaries, columns


def flag_chunk(chunk, boundaries):
    chunk = chunk.copy()
    flag_columns = []

    for column in METRICS:
        original = chunk[column]
        values = pd.to_numeric(original, errors="coerce")
        finite = np.isfinite(values)

        # A present but unreadable/infinite value is invalid.
        invalid = (original.notna() & ~finite)

        # Apply business rules before deciding what counts as an outlier.
        if column == "DaysOnMarket":
            invalid |= finite & (values < 0)
        else:
            invalid |= finite & (values <= 0)

        limits = boundaries[column]
        outlier = pd.Series(False, index=chunk.index)

        if limits is not None:
            outlier = (
                finite
                & ~invalid
                & (
                    (values < limits["lower"])
                    | (values > limits["upper"])
                )
            )

        invalid_name = f"{column}_invalid_flag"
        outlier_name = f"{column}_iqr_outlier_flag"

        chunk[invalid_name] = invalid
        chunk[outlier_name] = outlier

        flag_columns.extend([invalid_name, outlier_name])

    chunk["any_week7_flag"] = chunk[flag_columns].any(axis=1)
    return chunk


def process(label):
    source = OUTPUT / f"week6_{label}_features.csv"
    flagged_file = OUTPUT / f"week7_{label}_flagged.csv"
    filtered_file = OUTPUT / f"week7_{label}_filtered.csv"
    report_file = OUTPUT / f"week7_{label}_comparison.txt"

    if not source.exists():
        raise FileNotFoundError(f"Cannot find {source}")

    print(f"\nProcessing {label}...")
    boundaries, before_columns = get_boundaries(source)

    before_count = 0
    after_count = 0
    flag_counts = {
        f"{column}_{kind}_flag": 0
        for column in METRICS
        for kind in ["invalid", "iqr_outlier"]
    }

    for chunk_number, chunk in enumerate(
        pd.read_csv(source, chunksize=CHUNK_SIZE, low_memory=False)
    ):
        flagged = flag_chunk(chunk, boundaries)
        filtered = flagged.loc[~flagged["any_week7_flag"]].copy()

        flagged.to_csv(
            flagged_file,
            mode="w" if chunk_number == 0 else "a",
            header=chunk_number == 0,
            index=False,
        )
        filtered.to_csv(
            filtered_file,
            mode="w" if chunk_number == 0 else "a",
            header=chunk_number == 0,
            index=False,
        )

        before_count += len(flagged)
        after_count += len(filtered)

        for name in flag_counts:
            flag_counts[name] += int(flagged[name].sum())

        print(f"{label}: {before_count:,} rows checked", end="\r")

    # Read only three columns again to compare medians.
    after_columns = pd.read_csv(
        filtered_file, usecols=METRICS, low_memory=False
    )

    lines = [
        f"Week 7 comparison: {label}",
        f"Rows before filtering: {before_count:,}",
        f"Rows after filtering:  {after_count:,}",
        f"Rows flagged/removed from analysis copy: "
        f"{before_count - after_count:,}",
        "",
        "Median comparison (before -> after):",
    ]

    for column in METRICS:
        before_median = pd.to_numeric(
            before_columns[column], errors="coerce"
        ).median()
        after_median = pd.to_numeric(
            after_columns[column], errors="coerce"
        ).median()

        lines.append(
            f"{column}: {before_median:,.2f} -> {after_median:,.2f}"
        )

    lines.extend(["", "Flag counts:"])
    for name, count in flag_counts.items():
        lines.append(f"{name}: {count:,}")

    lines.extend(["", "IQR boundaries and reference percentiles:"])
    for column, limits in boundaries.items():
        if limits is None:
            lines.append(
                f"{column}: no valid values; IQR flag was not applied"
            )
        else:
            lines.append(
                f"{column}: lower={limits['lower']:,.2f}, "
                f"upper={limits['upper']:,.2f}, "
                f"1st percentile={limits['p01']:,.2f}, "
                f"99th percentile={limits['p99']:,.2f}"
            )

    report = "\n".join(lines)
    report_file.write_text(report + "\n", encoding="utf-8")

    print(f"\n{report}")
    print(f"\nSaved: {flagged_file}")
    print(f"Saved: {filtered_file}")
    print(f"Saved: {report_file}")


process("sold")
process("listings")
print("\nWeek 7 finished.")