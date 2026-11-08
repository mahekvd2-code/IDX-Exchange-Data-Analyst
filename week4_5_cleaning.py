from pathlib import Path
import pandas as pd

# Weeks 4–5: Data cleaning and preparation.
#
# Approach:
# - Preserve source files and all input rows.
# - Convert dates and numeric fields to appropriate types.
# - Flag invalid values, then replace invalid numeric cells with null.
# - Leave missing values unfilled unless there is a justified rule.
# - Remove only entirely empty, non-core columns.
# - Preserve repeated columns until their contents are reviewed.
# - Flag geographic issues without changing coordinates.
# - Save cleaned datasets and reports locally.

OUTPUT = Path("output")
REPORTS = OUTPUT / "week4_reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DATE_FIELDS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

NUMERIC_FIELDS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt",
    "Latitude",
    "Longitude",
    "rate_30yr_fixed",
]

# Retain these even when entirely missing.
CORE_FIELDS = set(DATE_FIELDS + NUMERIC_FIELDS + [
    "ListingKey",
    "ListingKeyNumeric",
    "ListingId",
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "City",
    "PostalCode",
    "StateOrProvince",
    "MLSAreaMajor",
    "MlsStatus",
    "ListAgentFullName",
    "ListAgentFirstName",
    "ListAgentLastName",
    "BuyerAgentFirstName",
    "BuyerAgentLastName",
    "BuyerAgentMlsId",
    "ListOfficeName",
    "BuyerOfficeName",
    "year_month",
])


def parse_dates(series):
    """Handle mixed date formats by parsing unique values individually."""
    mapping = {
        value: pd.to_datetime(value, errors="coerce", utc=True)
        for value in series.dropna().unique()
    }
    return pd.to_datetime(series.map(mapping), utc=True)


for name in ["sold", "listings"]:
    print(f"\nCleaning {name}...")

    # Keep identifiers as text, including postal-code leading zeros.
    text_fields = {
        "PostalCode": "string",
        "ListingId": "string",
        "ListingKey": "string",
        "ListingKeyNumeric": "string",
    }

    df = pd.read_csv(
        OUTPUT / f"week3_{name}_with_rates.csv",
        dtype=text_fields,
        low_memory=False,
    )

    rows_before = len(df)
    columns_before = df.shape[1]
    types_before = df.dtypes.astype(str)
    missing_before = df.isna().sum()
    changes = []

    assert df["PropertyType"].eq("Residential").all(), (
        "Unexpected property types: review the input file."
    )

    # -----------------------------------------------------
    # 1. Convert date fields and flag parsing failures
    # -----------------------------------------------------

    for field in DATE_FIELDS:
        original = df[field]
        parsed = parse_dates(original)

        flag = f"{field}_parse_error_flag"
        df[flag] = original.notna() & parsed.isna()
        df[field] = parsed

        changes.append({
            "field": field,
            "action": "Convert to UTC datetime; unreadable values become null",
            "affected_count": int(df[flag].sum()),
            "reason": "Enable date comparisons and identify parsing failures",
        })

    # -----------------------------------------------------
    # 2. Convert numeric fields and flag parsing failures
    # -----------------------------------------------------

    for field in NUMERIC_FIELDS:
        original = df[field]
        converted = pd.to_numeric(original, errors="coerce")

        parse_flag = f"{field}_numeric_parse_error_flag"
        df[parse_flag] = original.notna() & converted.isna()

        infinite_flag = f"{field}_infinite_flag"
        df[infinite_flag] = converted.isin([
            float("inf"), float("-inf")
        ])

        df[field] = converted.mask(df[infinite_flag])

        changes.append({
            "field": field,
            "action": "Convert to numeric; unreadable or infinite values become null",
            "affected_count": int(
                (df[parse_flag] | df[infinite_flag]).sum()
            ),
            "reason": "Ensure numeric calculations use finite numeric values",
        })

    # -----------------------------------------------------
    # 3. Flag invalid numeric values, then set those cells null
    # -----------------------------------------------------

    invalid_rules = {
        "ClosePrice": df["ClosePrice"] <= 0,
        "ListPrice": df["ListPrice"] <= 0,
        "OriginalListPrice": df["OriginalListPrice"] <= 0,
        "LivingArea": df["LivingArea"] <= 0,
        "DaysOnMarket": df["DaysOnMarket"] < 0,
        "BedroomsTotal": df["BedroomsTotal"] < 0,
        "BathroomsTotalInteger": df["BathroomsTotalInteger"] < 0,
    }

    for field, invalid in invalid_rules.items():
        flag = f"{field}_invalid_flag"
        df[flag] = invalid.fillna(False)

        # Preserve the row; only the invalid cell becomes null.
        df.loc[df[flag], field] = float("nan")

        changes.append({
            "field": field,
            "action": "Flag invalid value and replace cell with null",
            "affected_count": int(df[flag].sum()),
            "reason": (
                "Prices and living area must be positive; "
                "days on market, bedrooms, and bathrooms cannot be negative"
            ),
        })

    # -----------------------------------------------------
    # 4. Date consistency flags
    # Comparisons require both relevant dates to be available.
    # Missing dates are reported separately.
    # -----------------------------------------------------

    df["listing_after_close_flag"] = (
        df["ListingContractDate"] > df["CloseDate"]
    )
    df["purchase_after_close_flag"] = (
        df["PurchaseContractDate"] > df["CloseDate"]
    )
    df["negative_timeline_flag"] = (
        df["PurchaseContractDate"] < df["ListingContractDate"]
    )

    df["timeline_dates_missing_flag"] = df[
        ["ListingContractDate", "PurchaseContractDate", "CloseDate"]
    ].isna().any(axis=1)

    # -----------------------------------------------------
    # 5. Geographic quality flags
    # -----------------------------------------------------

    lat = df["Latitude"]
    lon = df["Longitude"]
    coordinates_present = lat.notna() & lon.notna()

    df["coordinates_missing_flag"] = ~coordinates_present
    df["coordinates_zero_flag"] = lat.eq(0) | lon.eq(0)
    df["positive_longitude_flag"] = lon.gt(0)

    df["coordinates_outside_world_flag"] = (
        (lat.notna() & ~lat.between(-90, 90))
        | (lon.notna() & ~lon.between(-180, 180))
    )

    # Broad California screening rectangle, not a state-boundary test.
    df["outside_california_screen_flag"] = (
        coordinates_present
        & ~(lat.between(32, 42.1) & lon.between(-124.6, -114))
    )

    state = (
        df["StateOrProvince"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    df["state_missing_flag"] = state.isna() | state.eq("")
    df["non_california_state_flag"] = (
        state.notna()
        & state.ne("")
        & ~state.isin(["CA", "CALIFORNIA"])
    ).fillna(False)

    df["geographic_review_flag"] = df[[
        "coordinates_missing_flag",
        "coordinates_zero_flag",
        "positive_longitude_flag",
        "coordinates_outside_world_flag",
        "outside_california_screen_flag",
        "non_california_state_flag",
    ]].any(axis=1)

    # -----------------------------------------------------
    # 6. Review repeated columns before deciding on removal
    # -----------------------------------------------------

    repeated_review = []

    for column in df.columns:
        if column.endswith(".1"):
            base = column[:-2]
            repeated_review.append({
                "column": column,
                "corresponding_column": base,
                "corresponding_column_exists": base in df.columns,
                "decision": "Retained pending comparison of source values",
            })

    pd.DataFrame(
        repeated_review,
        columns=[
            "column",
            "corresponding_column",
            "corresponding_column_exists",
            "decision",
        ],
    ).to_csv(
        REPORTS / f"{name}_repeated_column_review.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 7. Remove entirely empty, non-core columns only
    # -----------------------------------------------------

    drop_columns = [
        column for column in df.columns
        if column not in CORE_FIELDS
        and not column.endswith("_flag")
        and df[column].isna().all()
    ]

    for column in drop_columns:
        changes.append({
            "field": column,
            "action": "Drop entirely empty non-core column",
            "affected_count": len(df),
            "reason": "Contains no usable values and is not a required core field",
        })

    df = df.drop(columns=drop_columns)

    # No general imputation: missing does not automatically mean zero.
    changes.append({
        "field": "Other missing values",
        "action": "Retain as null",
        "affected_count": None,
        "reason": "Avoid inventing prices, sizes, dates, or property attributes",
    })

    # -----------------------------------------------------
    # 8. Save validation reports
    # -----------------------------------------------------

    flag_columns = [
        column for column in df.columns
        if column.endswith("_flag")
    ]

    flag_counts = df[flag_columns].sum().astype("int64")
    flag_counts.rename("flagged_rows").to_csv(
        REPORTS / f"{name}_flag_counts.csv"
    )

    geographic_flags = [
        "coordinates_missing_flag",
        "coordinates_zero_flag",
        "positive_longitude_flag",
        "coordinates_outside_world_flag",
        "outside_california_screen_flag",
        "state_missing_flag",
        "non_california_state_flag",
        "geographic_review_flag",
    ]

    df[geographic_flags].sum().astype("int64").rename(
        "flagged_rows"
    ).to_csv(REPORTS / f"{name}_geographic_summary.csv")

    type_report = pd.DataFrame({
        "before": types_before,
        "after": df.dtypes.astype(str),
    })
    type_report.index.name = "column"
    type_report.to_csv(REPORTS / f"{name}_data_types.csv")

    missing_report = pd.DataFrame({
        "missing_before": missing_before,
        "missing_after": df.isna().sum(),
    })
    missing_report.index.name = "column"
    missing_report.to_csv(
        REPORTS / f"{name}_missing_before_after.csv"
    )

    pd.DataFrame(changes).to_csv(
        REPORTS / f"{name}_transformation_log.csv",
        index=False,
    )

    # Summaries after handling invalid cells; extreme positive
    # values remain because statistical outlier handling is Week 7.
    df[[
        "ClosePrice", "LivingArea", "DaysOnMarket"
    ]].describe().to_csv(
        REPORTS / f"{name}_numeric_summary_after_cleaning.csv"
    )

    assert len(df) == rows_before, "Unexpected row-count change"

    for field in DATE_FIELDS:
        assert pd.api.types.is_datetime64_any_dtype(df[field])

    for field in NUMERIC_FIELDS:
        assert pd.api.types.is_numeric_dtype(df[field])

    assert df["rate_30yr_fixed"].notna().all(), (
        "Missing mortgage rates require review."
    )

    # -----------------------------------------------------
    # 9. Save a separate cleaned dataset
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT / f"week4_{name}_cleaned.csv",
        index=False,
    )

    pd.DataFrame([{
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_before": columns_before,
        "columns_after": df.shape[1],
        "dropped_empty_columns": len(drop_columns),
    }]).to_csv(
        REPORTS / f"{name}_row_column_counts.csv",
        index=False,
    )

    print("Rows before:", rows_before)
    print("Rows after:", len(df))
    print("Empty non-core columns removed:", drop_columns)
    print("Date consistency flags:")
    print(flag_counts[[
        "listing_after_close_flag",
        "purchase_after_close_flag",
        "negative_timeline_flag",
    ]].to_string())
    print("Rows needing geographic review:",
          int(df["geographic_review_flag"].sum()))
    print(f"Saved: output/week4_{name}_cleaned.csv")

    del df

print("\nCleaning finished. Review the reports before using the data.")