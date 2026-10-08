import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

input_file = "data/f1_model_dataset_v2.csv"

df = pd.read_csv(input_file)

print("Rows loaded:", len(df))


# ============================================================
# 2. LOAD MASTER DATA
# ============================================================

master_file = "data/f1_master_dataset_2018_2025.csv"

master = pd.read_csv(master_file)


# Fix constructor column if necessary

if "constructor_race" in master.columns:
    master["constructor"] = master["constructor_race"]


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

master = master.sort_values(
    ["season", "round"]
).reset_index(drop=True)


# ============================================================
# 4. DRIVER CHAMPIONSHIP POINTS BEFORE EACH RACE
# ============================================================

# Points from the current race must NOT be included.
#
# Example:
#
# Predict Round 10
#        ↓
# use points from Rounds 1–9
#
# NOT Round 10.


master["driver_points_before_race"] = (
    master
    .groupby(
        ["season", "driver"]
    )["points"]
    .transform(
        lambda x:
        x.shift(1)
        .fillna(0)
        .cumsum()
    )
)


# ============================================================
# 5. DRIVER CHAMPIONSHIP POSITION
# ============================================================

# We calculate championship ranking within each
# season and race.


driver_standings = (
    master[
        [
            "season",
            "round",
            "driver",
            "driver_points_before_race"
        ]
    ]
    .drop_duplicates()
    .copy()
)


driver_standings[
    "driver_championship_position"
] = (
    driver_standings
    .groupby(
        ["season", "round"]
    )["driver_points_before_race"]
    .rank(
        method="min",
        ascending=False
    )
)


# ============================================================
# 6. CONSTRUCTOR CHAMPIONSHIP POINTS
# ============================================================

constructor_race = (
    master
    .groupby(
        [
            "season",
            "round",
            "constructor"
        ],
        as_index=False
    )
    .agg(
        constructor_race_points=(
            "points",
            "sum"
        )
    )
)


constructor_race = (
    constructor_race
    .sort_values(
        [
            "constructor",
            "season",
            "round"
        ]
    )
)


# Points before each race

constructor_race[
    "constructor_points_before_race"
] = (
    constructor_race
    .groupby(
        ["season", "constructor"]
    )["constructor_race_points"]
    .transform(
        lambda x:
        x.shift(1)
        .fillna(0)
        .cumsum()
    )
)


# ============================================================
# 7. CONSTRUCTOR CHAMPIONSHIP POSITION
# ============================================================

constructor_race[
    "constructor_championship_position"
] = (
    constructor_race
    .groupby(
        ["season", "round"]
    )["constructor_points_before_race"]
    .rank(
        method="min",
        ascending=False
    )
)


# ============================================================
# 8. KEEP ONLY NEEDED DRIVER FEATURES
# ============================================================

driver_features = driver_standings[
    [
        "season",
        "round",
        "driver",
        "driver_points_before_race",
        "driver_championship_position"
    ]
]


# ============================================================
# 9. KEEP ONLY NEEDED CONSTRUCTOR FEATURES
# ============================================================

constructor_features = constructor_race[
    [
        "season",
        "round",
        "constructor",
        "constructor_points_before_race",
        "constructor_championship_position"
    ]
]


# ============================================================
# 10. MERGE DRIVER FEATURES
# ============================================================

df = df.merge(
    driver_features,
    on=[
        "season",
        "round",
        "driver"
    ],
    how="left"
)


# ============================================================
# 11. MERGE CONSTRUCTOR FEATURES
# ============================================================

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
# 12. SAVE FINAL FEATURE DATASET
# ============================================================

output_file = (
    "data/f1_model_dataset_v3.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 13. REPORT
# ============================================================

print("\n======================================")
print("CHAMPIONSHIP FEATURES COMPLETE")
print("======================================")

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


print("\nNew features:")

new_features = [
    "driver_points_before_race",
    "driver_championship_position",
    "constructor_points_before_race",
    "constructor_championship_position"
]

for feature in new_features:
    print("-", feature)


print("\nMissing values:")

print(
    df[new_features].isnull().sum()
)


print("\nSaved to:")

print(output_file)