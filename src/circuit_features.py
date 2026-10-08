import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

model_file = "data/f1_model_dataset.csv"
master_file = "data/f1_master_dataset_2018_2025.csv"

df = pd.read_csv(model_file)
master = pd.read_csv(master_file)

print("Model dataset rows:", len(df))


# ============================================================
# 2. FIX COLUMN NAMES CREATED DURING MERGE
# ============================================================

# Race name
if "race_race" in master.columns:
    master["race"] = master["race_race"]

elif "race" not in master.columns:
    raise ValueError(
        "Could not find race column."
    )


# Constructor
if "constructor_race" in master.columns:
    master["constructor"] = master["constructor_race"]

elif "constructor" not in master.columns:
    raise ValueError(
        "Could not find constructor column."
    )


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

master = master.sort_values(
    ["season", "round"]
).reset_index(drop=True)


# ============================================================
# 4. CREATE PODIUM TARGET
# ============================================================

master["podium"] = (
    master["position"] <= 3
).astype(int)


# ============================================================
# 5. IDENTIFY SINGAPORE RACES
# ============================================================

master["is_singapore"] = (
    master["race"]
    .str.lower()
    .str.contains("singapore")
)


print(
    "Singapore race records found:",
    master["is_singapore"].sum()
)


# ============================================================
# 6. DRIVER SINGAPORE HISTORY
# ============================================================

singapore = master[
    master["is_singapore"]
].copy()

singapore = singapore.sort_values(
    ["driver", "season", "round"]
)


# Previous Singapore average finish
singapore[
    "driver_singapore_avg_finish"
] = (
    singapore
    .groupby("driver")["position"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


# Previous Singapore podiums
singapore[
    "driver_singapore_podiums"
] = (
    singapore
    .groupby("driver")["podium"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .sum()
    )
)


# ============================================================
# 7. CONSTRUCTOR SINGAPORE HISTORY
# ============================================================

constructor_singapore = (
    singapore
    .groupby(
        ["season", "round", "constructor"],
        as_index=False
    )
    .agg(
        constructor_finish=(
            "position",
            "mean"
        ),

        constructor_podiums=(
            "podium",
            "sum"
        )
    )
)


constructor_singapore = (
    constructor_singapore
    .sort_values(
        ["constructor", "season", "round"]
    )
)


# Previous Singapore average finish
constructor_singapore[
    "constructor_singapore_avg_finish"
] = (
    constructor_singapore
    .groupby("constructor")
    ["constructor_finish"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


# Previous Singapore podiums
constructor_singapore[
    "constructor_singapore_podiums"
] = (
    constructor_singapore
    .groupby("constructor")
    ["constructor_podiums"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .sum()
    )
)


# ============================================================
# 8. DRIVER SINGAPORE FEATURES
# ============================================================

driver_singapore_features = singapore[
    [
        "season",
        "round",
        "driver",
        "driver_singapore_avg_finish",
        "driver_singapore_podiums"
    ]
]


# ============================================================
# 9. CONSTRUCTOR SINGAPORE FEATURES
# ============================================================

constructor_singapore_features = (
    constructor_singapore[
        [
            "season",
            "round",
            "constructor",
            "constructor_singapore_avg_finish",
            "constructor_singapore_podiums"
        ]
    ]
)


# ============================================================
# 10. MERGE SINGAPORE DRIVER FEATURES
# ============================================================

df = df.merge(
    driver_singapore_features,
    on=[
        "season",
        "round",
        "driver"
    ],
    how="left"
)


# ============================================================
# 11. MERGE SINGAPORE CONSTRUCTOR FEATURES
# ============================================================

df = df.merge(
    constructor_singapore_features,
    on=[
        "season",
        "round",
        "constructor"
    ],
    how="left"
)


# ============================================================
# 12. STREET CIRCUITS
# ============================================================

street_circuits = [
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Azerbaijan Grand Prix",
    "Saudi Arabian Grand Prix",
    "Las Vegas Grand Prix"
]


master["is_street_circuit"] = (
    master["race"].isin(street_circuits)
)


# ============================================================
# 13. DRIVER STREET HISTORY
# ============================================================

street = master[
    master["is_street_circuit"]
].copy()

street = street.sort_values(
    ["driver", "season", "round"]
)


street[
    "driver_street_avg_finish"
] = (
    street
    .groupby("driver")["position"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


# ============================================================
# 14. CONSTRUCTOR STREET HISTORY
# ============================================================

constructor_street = (
    street
    .groupby(
        [
            "season",
            "round",
            "constructor"
        ],
        as_index=False
    )
    .agg(
        constructor_street_finish=(
            "position",
            "mean"
        )
    )
)


constructor_street = (
    constructor_street
    .sort_values(
        ["constructor", "season", "round"]
    )
)


constructor_street[
    "constructor_street_avg_finish"
] = (
    constructor_street
    .groupby("constructor")
    ["constructor_street_finish"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


# ============================================================
# 15. MERGE STREET FEATURES
# ============================================================

driver_street_features = street[
    [
        "season",
        "round",
        "driver",
        "driver_street_avg_finish"
    ]
]


constructor_street_features = (
    constructor_street[
        [
            "season",
            "round",
            "constructor",
            "constructor_street_avg_finish"
        ]
    ]
)


df = df.merge(
    driver_street_features,
    on=[
        "season",
        "round",
        "driver"
    ],
    how="left"
)


df = df.merge(
    constructor_street_features,
    on=[
        "season",
        "round",
        "constructor"
    ],
    how="left"
)


# ============================================================
# 16. SAVE
# ============================================================

output_file = (
    "data/f1_model_dataset_v2.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 17. REPORT
# ============================================================

print("\n======================================")
print("CIRCUIT FEATURES COMPLETE")
print("======================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nNew features:")

new_features = [
    "driver_singapore_avg_finish",
    "driver_singapore_podiums",
    "constructor_singapore_avg_finish",
    "constructor_singapore_podiums",
    "driver_street_avg_finish",
    "constructor_street_avg_finish"
]

for feature in new_features:
    print("-", feature)

print("\nSaved to:")
print(output_file)