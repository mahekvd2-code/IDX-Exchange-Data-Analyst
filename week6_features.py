# Week 6: Create market metrics and match properties to school districts.
# Run from the IDX Exchange project folder: python week6_features.py

from pathlib import Path

import geopandas as gpd
import pandas as pd


OUTPUT = Path("output")
DISTRICT_FILE = Path("reference/school_districts.geojson")

if not DISTRICT_FILE.exists():
    raise FileNotFoundError(
        f"School district file not found: {DISTRICT_FILE}"
    )

# Public California school district boundary data.
districts = gpd.read_file(DISTRICT_FILE).to_crs("EPSG:4326")

needed = {"DistrictName", "DistrictType"}
missing = needed - set(districts.columns)
if missing:
    raise ValueError(f"District file is missing columns: {sorted(missing)}")

districts = districts[
    ["DistrictName", "DistrictType", "geometry"]
].dropna(subset=["geometry"]).copy()

districts["DistrictType"] = (
    districts["DistrictType"].fillna("").astype(str).str.lower()
)

# Elementary and high-school boundaries can overlap. Keep separate columns.
district_types = {
    "ElementaryDistrict": districts[
        districts["DistrictType"].str.contains("elementary")
    ],
    "HighSchoolDistrictBoundary": districts[
        districts["DistrictType"].str.contains("high")
    ],
    "UnifiedDistrict": districts[
        districts["DistrictType"].str.contains("unified")
    ],
}


def add_school_districts(data):
    data = data.copy()

    latitude = pd.to_numeric(data["Latitude"], errors="coerce")
    longitude = pd.to_numeric(data["Longitude"], errors="coerce")

    valid = latitude.between(-90, 90) & longitude.between(-180, 180)

    for column_name in district_types:
        data[column_name] = pd.NA

    if not valid.any():
        return data

    points = gpd.GeoDataFrame(
        {"row_number": data.index[valid]},
        geometry=gpd.points_from_xy(longitude[valid], latitude[valid]),
        crs="EPSG:4326",
    )

    for column_name, boundaries in district_types.items():
        if boundaries.empty:
            continue

        matches = gpd.sjoin(
            points,
            boundaries[["DistrictName", "geometry"]],
            how="left",
            predicate="within",
        )

        # Keep one match per property for this district type.
        matches = matches.drop_duplicates(subset=["row_number"])
        names = matches.set_index("row_number")["DistrictName"]
        data.loc[names.index, column_name] = names

    return data


def add_market_metrics(data):
    data = data.copy()

    for column in ["ClosePrice", "OriginalListPrice", "LivingArea"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in [
        "CloseDate",
        "ListingContractDate",
        "PurchaseContractDate",
    ]:
        data[column] = pd.to_datetime(
            data[column], errors="coerce", utc=True
        )

    # Avoid dividing by zero or a negative price/area.
    original_price = data["OriginalListPrice"].where(
        data["OriginalListPrice"] > 0
    )
    living_area = data["LivingArea"].where(data["LivingArea"] > 0)

    data["PriceRatio"] = data["ClosePrice"] / original_price
    data["CloseToOriginalListRatio"] = data["PriceRatio"]
    data["PricePerSqFt"] = data["ClosePrice"] / living_area

    # DaysOnMarket is already a column in the source data; retain it.
    data["ListingToContractDays"] = (
        data["PurchaseContractDate"] - data["ListingContractDate"]
    ).dt.days

    data["ContractToCloseDays"] = (
        data["CloseDate"] - data["PurchaseContractDate"]
    ).dt.days

    data["CloseYear"] = data["CloseDate"].dt.year.astype("Int64")
    data["CloseMonth"] = data["CloseDate"].dt.month.astype("Int64")
    data["CloseYrMo"] = data["CloseDate"].dt.strftime("%Y-%m")

    return add_school_districts(data)


def process_file(label):
    source = OUTPUT / f"week4_{label}_cleaned.csv"
    destination = OUTPUT / f"week6_{label}_features.csv"

    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    total_rows = 0

    # Process in chunks because the full CSVs are large.
    for chunk_number, chunk in enumerate(
        pd.read_csv(source, chunksize=10_000, low_memory=False)
    ):
        result = add_market_metrics(chunk.reset_index(drop=True))

        result.to_csv(
            destination,
            mode="w" if chunk_number == 0 else "a",
            header=chunk_number == 0,
            index=False,
        )

        total_rows += len(result)
        print(f"{label}: {total_rows:,} rows processed", end="\r")

    print(f"{label}: saved {total_rows:,} rows to {destination}")

    # Handbook requires at least one segmented summary.
    summary_data = pd.read_csv(
        destination,
        usecols=["CountyOrParish", "PriceRatio"],
        low_memory=False,
    )

    county_summary = (
        summary_data.groupby("CountyOrParish", dropna=False)
        .agg(
            PropertyCount=("PriceRatio", "size"),
            MedianPriceRatio=("PriceRatio", "median"),
        )
        .reset_index()
    )

    summary_file = OUTPUT / f"week6_{label}_county_summary.csv"
    county_summary.to_csv(summary_file, index=False)
    print(f"{label}: saved county summary to {summary_file}")


process_file("sold")
process_file("listings")

# Handbook also requires a sample showing the new columns populated.
sample_columns = [
    "DaysOnMarket",
    "PriceRatio",
    "PricePerSqFt",
    "CloseToOriginalListRatio",
    "CloseYear",
    "CloseMonth",
    "CloseYrMo",
    "ListingToContractDays",
    "ContractToCloseDays",
]

sample = (
    pd.read_csv(
        OUTPUT / "week6_sold_features.csv",
        usecols=sample_columns,
        nrows=1000,
    )
    .dropna()
    .head(5)
)

print("\nWeek 6 sample output:")
print(sample.to_string(index=False))

sample.to_csv(OUTPUT / "week6_sold_metric_sample.csv", index=False)

print("\nWeek 6 finished.")