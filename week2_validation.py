# Findings from validation:
# - Both Residential datasets contain 84 columns.
# - Each dataset has 15 columns above 90% missing,
#   including 8 columns that are entirely missing.
# - Sold data contains zero ClosePrice and LivingArea values,
#   and negative DaysOnMarket values.
# - Listings contain zero LivingArea and negative DaysOnMarket values.
# - Extreme price, living-area, and days-on-market values need review.
# - Listing ClosePrice statistics describe nonmissing values only.
# - No invalid values or outliers have been removed.

from pathlib import Path
import pandas as pd

# Weeks 2–3: Dataset structuring and validation.
# Inputs cover January 2024–September 2026.
# Week 1 already combined and filtered the Residential datasets.
# This script inspects original property categories, validates those
# filtered datasets, and saves missing-value and numeric reports.
# No columns or outliers are removed at this stage.

output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. Inspect property categories in the original monthly files
# ---------------------------------------------------------

for label, pattern in [
    ("Sold", "CRMLSSold*.csv"),
    ("Listings", "CRMLSListing*.csv")
]:
    files = sorted(Path("csv").glob(pattern))
    assert len(files) == 33, f"{label}: expected 33 monthly files"

    property_counts = pd.Series(dtype="int64")

    for file in files:
        table = pd.read_csv(file, usecols=["PropertyType"])
        counts = (
            table["PropertyType"]
            .fillna("(Missing)")
            .value_counts()
        )
        property_counts = property_counts.add(counts, fill_value=0)

    property_report = (
        property_counts
        .sort_values(ascending=False)
        .astype("int64")
        .rename("row_count")
        .to_frame()
    )
    property_report.index.name = "PropertyType"
    property_report["percent_of_rows"] = (
        property_report["row_count"]
        / property_report["row_count"].sum()
        * 100
    )

    property_report.to_csv(
        output_folder / f"week2_{label.lower()}_property_types.csv"
    )

    print(f"\n{label}: original property categories")
    print(property_report.to_string())

# ---------------------------------------------------------
# 2. Load the combined Residential datasets from Week 1
# ---------------------------------------------------------

sold = pd.read_csv(
    output_folder / "CRMLSSold_Residential_202401_202609.csv",
    low_memory=False
)

listings = pd.read_csv(
    output_folder / "CRMLSListing_Residential_202401_202609.csv",
    low_memory=False
)

# ---------------------------------------------------------
# 3. Validate, summarize, and save each dataset separately
# ---------------------------------------------------------

numeric_fields = ["ClosePrice", "LivingArea", "DaysOnMarket"]

for label, dataset in [("Sold", sold), ("Listings", listings)]:
    name = label.lower()

    print(f"\n{label}: dataset structure")
    print("Rows:", dataset.shape[0])
    print("Columns:", dataset.shape[1])

    structure_report = dataset.dtypes.astype(str).rename(
        "data_type"
    ).to_frame()
    structure_report.index.name = "column"

    structure_report.to_csv(
        output_folder / f"week2_{name}_structure.csv"
    )
    print(structure_report.to_string())

    # Apply the exact required filter again.
    # These inputs were already filtered during Week 1,
    # so the row count should remain unchanged.
    rows_before = len(dataset)
    residential = dataset[
        dataset["PropertyType"] == "Residential"
    ].copy()

    print("Rows before Residential filter:", rows_before)
    print("Rows after Residential filter:", len(residential))

    assert len(residential) == rows_before, (
        f"{label}: unexpected non-Residential or missing property types "
        "in the Week 1 filtered input"
    )

    # Missing-value report for every column.
    missing_report = pd.DataFrame({
        "missing_count": residential.isna().sum(),
        "missing_percent": residential.isna().mean() * 100
    })
    missing_report.index.name = "column"
    missing_report["above_90_percent"] = (
        missing_report["missing_percent"] > 90
    )
    missing_report = missing_report.sort_values(
        "missing_percent", ascending=False
    )

    missing_report.to_csv(
        output_folder / f"week2_{name}_missing_report.csv"
    )

    print(f"\n{label}: columns above 90% missing")
    print(
        missing_report[
            missing_report["above_90_percent"]
        ].to_string()
    )

    # Confirm the required summary fields are numeric.
    # Stop rather than silently converting unexpected values.
    for field in numeric_fields:
        assert pd.api.types.is_numeric_dtype(residential[field]), (
            f"{label}: {field} needs numeric-type review"
        )

    # Missing values are excluded from describe().
    # Zero, negative, and extreme values remain for inspection.
    numeric_summary = residential[numeric_fields].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    numeric_summary.index.name = "statistic"

    numeric_summary.to_csv(
        output_folder / f"week2_{name}_numeric_summary.csv"
    )

    print(f"\n{label}: numeric distribution summary")
    print(numeric_summary.to_string())

    residential.to_csv(
        output_folder / f"week2_{name}_residential.csv",
        index=False
    )

# Review decisions:
# - Columns above 90% missing are flagged as candidates for review.
# - Retain core market fields even when partially missing.
# - No columns are dropped until their purpose is reviewed.
# - Listing columns ending in .1 require comparison with their
#   corresponding unsuffixed columns before any removal.
# - The original listing categories included one "Resid" value
#   and one missing value; neither matches the exact Residential filter.
# - Date conversion, invalid-value handling, and further cleaning
#   are not performed by this script.

print("\nValidation complete. Reports and filtered CSVs saved in output.")