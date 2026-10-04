# Week 1: Combine January 2024–September 2026 data separately
# for sold properties and listings, then retain Residential rows.
#
# Sold: 33 monthly files.
# Rows before concatenation (sum): 736357; after: 736357.
# Rows before Residential filter: 736357; after: 495208.
#
# Listings: 33 monthly files.
# Rows before concatenation (sum): 984141; after: 984141.
# Rows before Residential filter: 984141; after: 626228.
#
# Final CSVs are saved locally in the output folder.

import pandas as pd
from pathlib import Path

sold_files = sorted(Path("csv").glob("CRMLSSold*.csv"))

print("Number of sold files:", len(sold_files))
print("First sold file:", sold_files[0].name)
print("Last sold file:", sold_files[-1].name)

sold_tables = []

for file in sold_files:
    table = pd.read_csv(file, low_memory=False)
    sold_tables.append(table)
    print(file.name, "rows:", len(table))

sold_rows_before = sum(len(table) for table in sold_tables)
print("Total sold rows before concatenation:", sold_rows_before)

sold_all = pd.concat(sold_tables, ignore_index=True)

print("Sold rows after concatenation:", len(sold_all))
assert len(sold_all) == sold_rows_before, "Row counts do not match"

print("Sold rows before Residential filter:", len(sold_all))

sold_residential = sold_all[sold_all["PropertyType"] == "Residential"].copy()

print("Sold rows after Residential filter:", len(sold_residential))

listing_files = sorted(Path("csv").glob("CRMLSListing*.csv"))
print("Number of listing files:", len(listing_files))

listing_tables = []

for file in listing_files:
    table = pd.read_csv(file, low_memory=False)
    listing_tables.append(table)

listing_rows_before = sum(len(table) for table in listing_tables)
listing_all = pd.concat(listing_tables, ignore_index=True)

print("Listing rows before concatenation:", listing_rows_before)
print("Listing rows after concatenation:", len(listing_all))

assert len(listing_all) == listing_rows_before, "Listing row counts do not match"

print("Listing rows before Residential filter:", len(listing_all))

listing_residential = listing_all[
    listing_all["PropertyType"] == "Residential"
].copy()

print("Listing rows after Residential filter:", len(listing_residential))

Path("output").mkdir(exist_ok=True)

sold_residential.to_csv("output/CRMLSSold_Residential_202401_202609.csv", index=False)

listing_residential.to_csv("output/CRMLSListing_Residential_202401_202609.csv", index=False)

print("Saved both Residential CSVs in the output folder.")