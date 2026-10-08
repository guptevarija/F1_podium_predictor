import pandas as pd
import numpy as np
import os

# ============================================================
# 2026 SINGAPORE GP
# PRE-QUALIFYING PREDICTION FEATURES
# ============================================================

HISTORICAL_FILE = "data/f1_race_results_2018_2025.csv"
CURRENT_FILE = "data/f1_race_results_2026.csv"

OUTPUT_FILE = "data/f1_singapore_2026_features.csv"


print("=" * 70)
print("2026 SINGAPORE GP - PRE-QUALIFYING FEATURES")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

historical = pd.read_csv(HISTORICAL_FILE)
current = pd.read_csv(CURRENT_FILE)

print("\nHistorical rows:", len(historical))
print("2026 rows:", len(current))


# ============================================================
# 2. BASIC CLEANING
# ============================================================

for df in [historical, current]:

    df["position"] = pd.to_numeric(
        df["position"],
        errors="coerce"
    )

    df["points"] = pd.to_numeric(
        df["points"],
        errors="coerce"
    )

    df["round"] = pd.to_numeric(
        df["round"],
        errors="coerce"
    )


# ============================================================
# 3. CREATE PODIUM + DNF
# ============================================================

historical["podium"] = (
    historical["position"] <= 3
).astype(int)

current["podium"] = (
    current["position"] <= 3
).astype(int)


historical["dnf"] = (
    ~historical["status"].astype(str).isin(
        ["Finished", "Lapped"]
    )
).astype(int)

current["dnf"] = (
    ~current["status"].astype(str).isin(
        ["Finished", "Lapped"]
    )
).astype(int)


# ============================================================
# 4. 2026 DRIVER RECENT FORM
# ============================================================

print("\nCalculating 2026 driver form...")

current = current.sort_values(
    ["driver", "round"]
).reset_index(drop=True)


driver_group = current.groupby("driver")


current["driver_avg_finish_last_5"] = (
    driver_group["position"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )
)


current["driver_avg_points_last_5"] = (
    driver_group["points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )
)


current["driver_podiums_last_5"] = (
    driver_group["podium"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .sum()
    )
)


current["driver_dnf_rate_last_5"] = (
    driver_group["dnf"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )
)

# ============================================================
# 5. 2026 CONSTRUCTOR RECENT FORM
# ============================================================

print("Calculating 2026 constructor form...")


# One row per constructor per race
constructor_races = (
    current
    .groupby(
        ["round", "constructor"],
        as_index=False
    )
    .agg(
        constructor_avg_finish=(
            "position",
            "mean"
        ),

        constructor_points=(
            "points",
            "sum"
        ),

        constructor_podiums=(
            "podium",
            "sum"
        ),

        constructor_dnf=(
            "dnf",
            "sum"
        )
    )
)


constructor_races = constructor_races.sort_values(
    ["constructor", "round"]
)


constructor_group = constructor_races.groupby(
    "constructor"
)


constructor_races[
    "constructor_avg_finish_last_5"
] = (
    constructor_group["constructor_avg_finish"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )
)


constructor_races[
    "constructor_avg_points_last_5"
] = (
    constructor_group["constructor_points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )
)


constructor_races[
    "constructor_podiums_last_5"
] = (
    constructor_group["constructor_podiums"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .sum()
    )
)


constructor_races[
    "constructor_dnf_rate_last_5"
] = (
    constructor_group["constructor_dnf"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )
)


# ============================================================
# 6. SINGAPORE HISTORY — 2018 TO 2025 ONLY
# ============================================================

print("Calculating Singapore history...")


singapore = historical[
    historical["race"]
    .astype(str)
    .str.contains(
        "Singapore",
        case=False,
        na=False
    )
].copy()


# Driver Singapore history
singapore_driver = (
    singapore
    .groupby("driver")
    .agg(
        driver_singapore_avg_finish=(
            "position",
            "mean"
        ),

        driver_singapore_podiums=(
            "podium",
            "sum"
        )
    )
    .reset_index()
)


# Constructor Singapore history
singapore_constructor = (
    singapore
    .groupby("constructor")
    .agg(
        constructor_singapore_avg_finish=(
            "position",
            "mean"
        ),

        constructor_singapore_podiums=(
            "podium",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# 7. STREET CIRCUIT HISTORY
# ============================================================

print("Calculating street circuit history...")


street_pattern = (
    "Singapore|Monaco|Baku|"
    "Jeddah|Las Vegas|Miami|Melbourne"
)


street = historical[
    historical["race"]
    .astype(str)
    .str.contains(
        street_pattern,
        case=False,
        regex=True,
        na=False
    )
].copy()


street_driver = (
    street
    .groupby("driver")
    .agg(
        driver_street_avg_finish=(
            "position",
            "mean"
        )
    )
    .reset_index()
)


street_constructor = (
    street
    .groupby("constructor")
    .agg(
        constructor_street_avg_finish=(
            "position",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 8. 2026 CHAMPIONSHIP
# ============================================================

print("Calculating 2026 championship...")


# Driver points after Round 16
driver_championship = (
    current
    .groupby("driver")
    .agg(
        driver_points_before_race=(
            "points",
            "sum"
        )
    )
    .reset_index()
)


driver_championship[
    "driver_championship_position"
] = (
    driver_championship[
        "driver_points_before_race"
    ]
    .rank(
        method="min",
        ascending=False
    )
)


# Constructor points after Round 16
constructor_championship = (
    current
    .groupby("constructor")
    .agg(
        constructor_points_before_race=(
            "points",
            "sum"
        )
    )
    .reset_index()
)


constructor_championship[
    "constructor_championship_position"
] = (
    constructor_championship[
        "constructor_points_before_race"
    ]
    .rank(
        method="min",
        ascending=False
    )
)


# ============================================================
# 9. GET LATEST DRIVER RECORD
# ============================================================

latest_driver = (
    current
    .sort_values("round")
    .groupby("driver")
    .tail(1)
    .copy()
)


# ============================================================
# 10. KEEP ONLY THE FEATURES WE NEED
# ============================================================

prediction = latest_driver[
    [
        "driver",
        "constructor"
    ]
].copy()


# ============================================================
# 11. MERGE DRIVER RECENT FORM
# ============================================================

driver_features = latest_driver[
    [
        "driver",

        "driver_avg_finish_last_5",
        "driver_avg_points_last_5",
        "driver_podiums_last_5",
        "driver_dnf_rate_last_5"
    ]
].copy()


prediction = prediction.merge(
    driver_features,
    on="driver",
    how="left"
)


# ============================================================
# 12. MERGE CONSTRUCTOR FORM
# ============================================================

prediction = prediction.merge(
    constructor_races[
        [
            "constructor",

            "constructor_avg_finish_last_5",
            "constructor_avg_points_last_5",
            "constructor_podiums_last_5",
            "constructor_dnf_rate_last_5"
        ]
    ]
    .drop_duplicates("constructor"),
    
    on="constructor",
    how="left"
)


# ============================================================
# 13. MERGE SINGAPORE HISTORY
# ============================================================

prediction = prediction.merge(
    singapore_driver,
    on="driver",
    how="left"
)


prediction = prediction.merge(
    singapore_constructor,
    on="constructor",
    how="left"
)


# ============================================================
# 14. MERGE STREET HISTORY
# ============================================================

prediction = prediction.merge(
    street_driver,
    on="driver",
    how="left"
)


prediction = prediction.merge(
    street_constructor,
    on="constructor",
    how="left"
)


# ============================================================
# 15. MERGE CHAMPIONSHIP
# ============================================================

prediction = prediction.merge(
    driver_championship,
    on="driver",
    how="left"
)


prediction = prediction.merge(
    constructor_championship,
    on="constructor",
    how="left"
)


# ============================================================
# 16. FINAL FEATURE LIST
# ============================================================

features = [

    "driver_avg_finish_last_5",
    "driver_avg_points_last_5",
    "driver_podiums_last_5",
    "driver_dnf_rate_last_5",

    "constructor_avg_finish_last_5",
    "constructor_avg_points_last_5",
    "constructor_podiums_last_5",
    "constructor_dnf_rate_last_5",

    "driver_singapore_avg_finish",
    "driver_singapore_podiums",

    "constructor_singapore_avg_finish",
    "constructor_singapore_podiums",

    "driver_street_avg_finish",
    "constructor_street_avg_finish",

    "driver_points_before_race",
    "driver_championship_position",

    "constructor_points_before_race",
    "constructor_championship_position"
]


# ============================================================
# 17. CHECK FEATURES
# ============================================================

print("\nChecking features...")

for feature in features:

    if feature not in prediction.columns:

        print(
            "WARNING - Missing:",
            feature
        )


# ============================================================
# 18. SAVE
# ============================================================

os.makedirs(
    "data",
    exist_ok=True
)

prediction.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS!")
print("=" * 70)

print(
    "\nNumber of drivers:",
    len(prediction)
)

print(
    "\nDrivers entering Singapore:"
)

print(
    prediction[
        [
            "driver",
            "constructor"
        ]
    ].to_string(index=False)
)


print(
    "\nFeature count:",
    len(features)
)


print(
    "\nSaved file:"
)

print(
    OUTPUT_FILE
)


print("\nFeature preview:")

print(
    prediction[
        ["driver"] + features
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("2026 FEATURE BUILD COMPLETE")
print("=" * 70)