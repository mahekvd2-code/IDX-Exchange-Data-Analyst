from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Complete the remaining Weeks 2–3 analysis.
# Original CSVs are unchanged. Reports and plots stay local.
# Existing Week 2 code provides property-type and missing-value reports.

output = Path("output")
reports = output / "week3_reports"
reports.mkdir(exist_ok=True)

fields = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres", "BedroomsTotal",
    "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt"
]

# Fetch weekly mortgage rates and calculate calendar-month averages.
rates = pd.read_csv(
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
)
rates["year_month"] = pd.to_datetime(
    rates["observation_date"]
).dt.to_period("M")
rates["rate_30yr_fixed"] = pd.to_numeric(
    rates["MORTGAGE30US"], errors="coerce"
)
monthly_rates = rates.groupby(
    "year_month", as_index=False
)["rate_30yr_fixed"].mean()

monthly_rates.to_csv(reports / "monthly_mortgage_rates.csv", index=False)

validation = []

for name in ["sold", "listings"]:
    print(f"\nProcessing {name}...")

    df = pd.read_csv(
        output / f"week2_{name}_residential.csv",
        low_memory=False
    )
    assert df["PropertyType"].eq("Residential").all()

    # Numeric summaries and plots: no records removed.
    numbers = df[fields].apply(pd.to_numeric, errors="coerce")
    numbers.describe(
        percentiles=[.01, .05, .25, .50, .75, .95, .99]
    ).to_csv(reports / f"{name}_numeric_summary.csv")

    for field in fields:
        values = numbers[field].dropna()
        values = values[
            ~values.isin([float("inf"), float("-inf")])
        ]
        if values.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        values.plot.hist(bins=40, ax=axes[0])
        axes[0].set_title("Full range")
        axes[1].boxplot(values.to_numpy(), vert=False)
        axes[1].set_title("Full-range boxplot")

        # Separate zoomed view; source data is not changed.
        middle = values[
            values.between(values.quantile(.01), values.quantile(.99))
        ]
        middle.plot.hist(bins=40, ax=axes[2])
        axes[2].set_title("1st–99th percentile view")
        fig.suptitle(f"{name}: {field}, before cleaning")
        fig.tight_layout()
        fig.savefig(reports / f"{name}_{field}.png")
        plt.close(fig)

    # Initial field decisions; no columns are dropped.
    core = set(fields + [
        "PropertyType", "PropertySubType", "CloseDate",
        "ListingContractDate", "PurchaseContractDate",
        "ContractStatusChangeDate", "Latitude", "Longitude",
        "CountyOrParish", "City", "PostalCode", "MLSAreaMajor",
        "MlsStatus", "ListOfficeName", "BuyerOfficeName",
        "ListAgentFullName", "ListAgentFirstName", "ListAgentLastName",
        "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerAgentMlsId"
    ])
    metadata = {
        "ListingKey", "ListingKeyNumeric", "ListingId",
        "OriginatingSystemName", "OriginatingSystemSubName",
        "latfilled", "lonfilled"
    }
    review = pd.DataFrame({"missing_percent": df.isna().mean() * 100})
    review["group"] = [
        "Market analysis / validation" if c in core
        else "Metadata" if c in metadata else "Supporting / review"
        for c in review.index
    ]
    review["decision"] = [
        "Retain core field" if c in core
        else "Compare repeated field before removal" if c.endswith(".1")
        else "Propose drop: sparse non-core field"
        if review.loc[c, "missing_percent"] > 90
        else "Retain for now"
        for c in review.index
    ]
    review.to_csv(reports / f"{name}_field_review.csv")

    # Parse dates for inspection and the monthly join.
    dates = {}
    for field in ["CloseDate", "ListingContractDate", "PurchaseContractDate"]:
        # Parsing unique values individually supports mixed date formats.
        mapping = {
            value: pd.to_datetime(value, errors="coerce", utc=True)
            for value in df[field].dropna().unique()
        }
        dates[field] = pd.to_datetime(df[field].map(mapping), utc=True)

    date_checks = {
        "Listing after close": (
            dates["ListingContractDate"] > dates["CloseDate"]
        ).sum(),
        "Purchase after close": (
            dates["PurchaseContractDate"] > dates["CloseDate"]
        ).sum(),
        "Listing after purchase": (
            dates["ListingContractDate"] > dates["PurchaseContractDate"]
        ).sum()
    }
    pd.Series(date_checks, name="count").to_csv(
        reports / f"{name}_date_checks.csv"
    )

    # Sale-price questions use sold records.
    if name == "sold":
        print("Mean close price:", numbers["ClosePrice"].mean())
        print("Median close price:", numbers["ClosePrice"].median())
        print("Median days on market:", numbers["DaysOnMarket"].median())

        valid = numbers["ClosePrice"].gt(0) & numbers["ListPrice"].gt(0)
        close = numbers.loc[valid, "ClosePrice"]
        listed = numbers.loc[valid, "ListPrice"]
        comparison = pd.Series({
            "Above list": (close > listed).sum(),
            "At list": (close == listed).sum(),
            "Below list": (close < listed).sum()
        }).to_frame("count")
        comparison["percent"] = (
            comparison["count"] / valid.sum() * 100
            if valid.any() else float("nan")
        )
        comparison.to_csv(reports / "sold_vs_list_price.csv")

        df.groupby("CountyOrParish")["ClosePrice"].agg(
            ["count", "median", "mean"]
        ).sort_values("median", ascending=False).to_csv(
            reports / "county_prices.csv"
        )

    # Join each property to its month's average mortgage rate.
    date_field = "CloseDate" if name == "sold" else "ListingContractDate"
    df["year_month"] = (
        dates[date_field].dt.tz_localize(None).dt.to_period("M")
    )
    enriched = df.merge(
        monthly_rates, on="year_month", how="left",
        validate="many_to_one"
    )
    assert len(enriched) == len(df), "Merge changed the row count"

    missing = int(enriched["rate_30yr_fixed"].isna().sum())
    validation.append({
        "dataset": name,
        "rows_before": len(df),
        "rows_after": len(enriched),
        "null_rates": missing
    })

    if missing:
        enriched.loc[
            enriched["rate_30yr_fixed"].isna(), "year_month"
        ].astype(str).value_counts().to_csv(
            reports / f"{name}_unmatched_months.csv"
        )
        print(f"{name}: {missing} unmatched rates. Needs review.")
    else:
        enriched.to_csv(
            output / f"week3_{name}_with_rates.csv", index=False
        )
        print(f"{name}: enriched CSV saved; no missing rates.")

pd.DataFrame(validation).to_csv(
    reports / "merge_validation.csv", index=False
)

assert all(item["null_rates"] == 0 for item in validation), (
    "Some mortgage rates did not match. Check unmatched-month reports."
)
print("\nFinished: reports, plots, and both enriched datasets saved.")