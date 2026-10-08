import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/f1_master_dataset_2018_2025.csv"

df = pd.read_csv(file_path)

print("Original rows:", len(df))


# ============================================================
# 2. HANDLE CONSTRUCTOR COLUMN
# ============================================================

# When we merged race and qualifying data, Pandas may have
# created constructor_race and constructor_qualifying.

if "constructor_race" in df.columns:
    df["constructor"] = df["constructor_race"]

elif "constructor" not in df.columns:
    raise ValueError(
        "Constructor column not found in the dataset."
    )


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    ["season", "round"]
).reset_index(drop=True)


# ============================================================
# 4. CREATE TARGET VARIABLE
# ============================================================

# 1 = podium
# 0 = no podium

df["podium"] = (
    df["position"] <= 3
).astype(int)


# ============================================================
# 5. CREATE NUMERICAL DNF VARIABLE
# ============================================================

# F1 status contains text such as:
# Finished
# Lapped
# Retired
# Accident
# Collision
# Engine
#
# We convert it to:
# 0 = completed/classified
# 1 = did not finish normally

df["dnf"] = (
    ~df["status"].isin(
        ["Finished", "Lapped"]
    )
).astype(int)


# ============================================================
# 6. DRIVER RECENT FORM
# ============================================================

# IMPORTANT:
# shift(1) means the current race is excluded.
#
# Example:
#
# Predicting Race 10
#       ↓
# use Races 5,6,7,8,9
#
# NOT Race 10.
#
# This prevents data leakage.


df["driver_avg_finish_last_5"] = (
    df.groupby("driver")["position"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .mean()
    )
)


df["driver_avg_points_last_5"] = (
    df.groupby("driver")["points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .mean()
    )
)


df["driver_podiums_last_5"] = (
    df.groupby("driver")["podium"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .sum()
    )
)


df["driver_dnf_rate_last_5"] = (
    df.groupby("driver")["dnf"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .mean()
    )
)


# ============================================================
# 7. CONSTRUCTOR RECENT FORM
# ============================================================

# IMPORTANT:
#
# We first create ONE observation per constructor per race.
#
# This prevents a constructor with two drivers from being
# counted twice when calculating team form.


constructor_race = (
    df.groupby(
        ["season", "round", "constructor"],
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

        constructor_dnf_rate=(
            "dnf",
            "mean"
        )
    )
)


# Sort constructor history chronologically

constructor_race = constructor_race.sort_values(
    ["constructor", "season", "round"]
)


# ------------------------------------------------------------
# Constructor average finish — previous 5 races
# ------------------------------------------------------------

constructor_race[
    "constructor_avg_finish_last_5"
] = (
    constructor_race
    .groupby("constructor")
    ["constructor_avg_finish"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .mean()
    )
)


# ------------------------------------------------------------
# Constructor points — previous 5 races
# ------------------------------------------------------------

constructor_race[
    "constructor_avg_points_last_5"
] = (
    constructor_race
    .groupby("constructor")
    ["constructor_points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .mean()
    )
)


# ------------------------------------------------------------
# Constructor podiums — previous 5 races
# ------------------------------------------------------------

constructor_race[
    "constructor_podiums_last_5"
] = (
    constructor_race
    .groupby("constructor")
    ["constructor_podiums"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .sum()
    )
)


# ------------------------------------------------------------
# Constructor DNF rate — previous 5 races
# ------------------------------------------------------------

constructor_race[
    "constructor_dnf_rate_last_5"
] = (
    constructor_race
    .groupby("constructor")
    ["constructor_dnf_rate"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=5,
            min_periods=5
        )
        .mean()
    )
)


# ============================================================
# 8. MERGE CONSTRUCTOR FEATURES BACK
# ============================================================

constructor_features = constructor_race[
    [
        "season",
        "round",
        "constructor",

        "constructor_avg_finish_last_5",

        "constructor_avg_points_last_5",

        "constructor_podiums_last_5",

        "constructor_dnf_rate_last_5"
    ]
]


df = df.merge(
    constructor_features,
    on=[
        "season",
        "round",
        "constructor"
    ],
    how="left"
)


# ============================================================
# 9. EXISTING QUALIFYING FEATURES
# ============================================================

# qualifying_position comes from our qualifying dataset.
#
# This model will represent a prediction made AFTER qualifying
# and BEFORE the race.
#
# Therefore qualifying position and grid position are valid
# predictive features.


if "qualifying_position" in df.columns:

    print(
        "\nQualifying data successfully found."
    )

else:

    print(
        "\nWARNING: qualifying_position "
        "not found."
    )


# ============================================================
# 10. DEFINE MODEL FEATURES
# ============================================================

feature_columns = [

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
    "constructor_dnf_rate_last_5"
]


# ============================================================
# 11. CHECK WHICH FEATURES EXIST
# ============================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]


if missing_features:

    print(
        "\nERROR: Missing features:"
    )

    for feature in missing_features:
        print("-", feature)

    raise ValueError(
        "Some required features are missing."
    )


# ============================================================
# 12. REMOVE ROWS WITHOUT ENOUGH HISTORY
# ============================================================

# The first few races for a driver/team don't have 5 previous
# races, so those rows cannot have complete rolling features.


df_model = df.dropna(
    subset=feature_columns
).copy()


# ============================================================
# 13. FINAL DATASET
# ============================================================

output_file = (
    "data/f1_model_dataset.csv"
)

df_model.to_csv(
    output_file,
    index=False
)


# ============================================================
# 14. REPORT
# ============================================================

print("\n======================================")
print("FEATURE ENGINEERING COMPLETE")
print("======================================")

print(
    f"Original rows: {len(df)}"
)

print(
    f"Model rows: {len(df_model)}"
)

print(
    f"Rows removed: "
    f"{len(df) - len(df_model)}"
)

print(
    f"Number of features: "
    f"{len(feature_columns)}"
)


print("\nMODEL FEATURES:")

for feature in feature_columns:
    print(f"- {feature}")


print("\nTARGET:")

print("podium")


print("\nTARGET DISTRIBUTION:")

print(
    df_model["podium"]
    .value_counts()
)


print("\nMISSING VALUES IN MODEL FEATURES:")

print(
    df_model[feature_columns]
    .isnull()
    .sum()
)


print("\nSaved to:")

print(output_file)