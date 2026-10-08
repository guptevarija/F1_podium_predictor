import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/f1_model_dataset_v3.csv"

df = pd.read_csv(file_path)

print("Total rows:", len(df))


# ============================================================
# 2. FIX MERGED COLUMN NAMES
# ============================================================

if "race_race" in df.columns:
    df["race"] = df["race_race"]

if "constructor_race" in df.columns:
    df["constructor"] = df["constructor_race"]


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

features = [

    "qualifying_position",
    "grid",

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


target = "podium"


# ============================================================
# 4. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print("-", feature)

    raise ValueError(
        "Required features are missing."
    )


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Some Singapore-history features are naturally missing
# for drivers who have never raced in Singapore before.
#
# We will use the median of the TRAINING data later.
#
# For now, keep the missing values.


# ============================================================
# 6. CREATE TIME-BASED SPLITS
# ============================================================

train = df[
    df["season"] <= 2023
].copy()

validation = df[
    df["season"] == 2024
].copy()

test = df[
    df["season"] == 2025
].copy()


# ============================================================
# 7. CREATE X AND Y
# ============================================================

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]


# ============================================================
# 8. PRINT SPLIT INFORMATION
# ============================================================

print("\n======================================")
print("TIME-BASED DATA SPLIT")
print("======================================")


print("\nTRAIN")
print("Years: 2018–2023")
print("Rows:", len(train))
print("Podiums:", y_train.sum())


print("\nVALIDATION")
print("Year: 2024")
print("Rows:", len(validation))
print("Podiums:", y_validation.sum())


print("\nTEST")
print("Year: 2025")
print("Rows:", len(test))
print("Podiums:", y_test.sum())


# ============================================================
# 9. CHECK TARGET BALANCE
# ============================================================

print("\n--- PODIUM RATE ---")

print(
    "Train:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Validation:",
    round(y_validation.mean() * 100, 2),
    "%"
)

print(
    "Test:",
    round(y_test.mean() * 100, 2),
    "%"
)


# ============================================================
# 10. SAVE SPLITS
# ============================================================

train.to_csv(
    "data/train.csv",
    index=False
)

validation.to_csv(
    "data/validation.csv",
    index=False
)

test.to_csv(
    "data/test.csv",
    index=False
)


print("\nSaved:")

print("data/train.csv")
print("data/validation.csv")
print("data/test.csv")


print("\n======================================")
print("DATA PREPARATION COMPLETE")
print("======================================")