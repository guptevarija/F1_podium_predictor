import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/f1_model_dataset_v3.csv"

df = pd.read_csv(file_path)


# ============================================================
# 2. FIX MERGED COLUMN NAMES
# ============================================================

# Race name
if "race_race" in df.columns:
    df["race"] = df["race_race"]

elif "race" not in df.columns:
    raise ValueError(
        "Race column not found."
    )


# Constructor
if "constructor_race" in df.columns:
    df["constructor"] = df["constructor_race"]

elif "constructor" not in df.columns:
    raise ValueError(
        "Constructor column not found."
    )


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n======================================")
print("F1 DATASET AUDIT")
print("======================================")

print("\n--- BASIC INFORMATION ---")

print("Rows:", len(df))
print("Columns:", len(df.columns))


print("\nColumn names:")

for column in df.columns:
    print("-", column)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n--- TARGET DISTRIBUTION ---")

print(
    df["podium"].value_counts()
)


print("\nPodium percentage:")

print(
    round(df["podium"].mean() * 100, 2),
    "%"
)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n--- MISSING VALUES ---")

missing = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")

else:
    print(missing)


# ============================================================
# 6. DUPLICATES
# ============================================================

print("\n--- DUPLICATES ---")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# 7. SEASONS
# ============================================================

print("\n--- SEASONS ---")

print(
    df["season"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. RACES PER SEASON
# ============================================================

print("\n--- RACES PER SEASON ---")

races = (
    df.groupby("season")["round"]
    .nunique()
)

print(races)


# ============================================================
# 9. SINGAPORE RACES
# ============================================================

print("\n--- SINGAPORE RACES ---")

singapore = df[
    df["race"]
    .str.lower()
    .str.contains("singapore")
]

singapore_races = (
    singapore[
        ["season", "round", "race"]
    ]
    .drop_duplicates()
    .sort_values(
        ["season", "round"]
    )
)

print(singapore_races)


# ============================================================
# 10. DATA TYPES
# ============================================================

print("\n--- DATA TYPES ---")

print(df.dtypes)


# ============================================================
# 11. MODEL FEATURES
# ============================================================

features = [

    # Qualifying / starting position
    "qualifying_position",
    "grid",

    # Driver recent form
    "driver_avg_finish_last_5",
    "driver_avg_points_last_5",
    "driver_podiums_last_5",
    "driver_dnf_rate_last_5",

    # Constructor recent form
    "constructor_avg_finish_last_5",
    "constructor_avg_points_last_5",
    "constructor_podiums_last_5",
    "constructor_dnf_rate_last_5",

    # Singapore history
    "driver_singapore_avg_finish",
    "driver_singapore_podiums",
    "constructor_singapore_avg_finish",
    "constructor_singapore_podiums",

    # Street circuits
    "driver_street_avg_finish",
    "constructor_street_avg_finish",

    # Championship
    "driver_points_before_race",
    "driver_championship_position",
    "constructor_points_before_race",
    "constructor_championship_position"
]


print("\n--- MODEL FEATURES ---")


missing_features = []


for feature in features:

    if feature in df.columns:

        print(
            f"{feature}: "
            f"OK | Missing = "
            f"{df[feature].isnull().sum()}"
        )

    else:

        print(
            f"{feature}: MISSING!"
        )

        missing_features.append(feature)


# ============================================================
# 12. FEATURE SUMMARY
# ============================================================

print("\n--- FEATURE SUMMARY ---")

existing_features = [
    feature
    for feature in features
    if feature in df.columns
]


print(
    df[existing_features]
    .describe()
    .T[
        [
            "min",
            "max",
            "mean",
            "std"
        ]
    ]
)


# ============================================================
# 13. SAMPLE PODIUM ROWS
# ============================================================

print("\n--- SAMPLE PODIUM ROWS ---")

podium_columns = [
    "season",
    "round",
    "race",
    "driver",
    "constructor",
    "qualifying_position",
    "grid",
    "driver_avg_finish_last_5",
    "constructor_avg_finish_last_5",
    "driver_championship_position",
    "podium"
]


print(
    df[
        df["podium"] == 1
    ][podium_columns]
    .head(10)
)


# ============================================================
# 14. SINGAPORE FEATURE CHECK
# ============================================================

print("\n--- SINGAPORE FEATURE CHECK ---")

singapore_check = df[
    df["race"]
    .str.lower()
    .str.contains("singapore")
]


singapore_columns = [
    "season",
    "driver",
    "qualifying_position",
    "driver_singapore_avg_finish",
    "driver_singapore_podiums",
    "driver_street_avg_finish",
    "driver_championship_position",
    "podium"
]


print(
    singapore_check[
        singapore_columns
    ]
    .head(20)
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("AUDIT COMPLETE")
print("======================================")

print(
    "Total rows:",
    len(df)
)

print(
    "Total columns:",
    len(df.columns)
)

print(
    "Total model features available:",
    len(existing_features)
)

print(
    "Missing model features:",
    len(missing_features)
)

if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print("-", feature)

else:

    print(
        "\nAll expected model features are present!"
    )