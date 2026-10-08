import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# ============================================================
# 2. FEATURES
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


X_train = train[features]
y_train = train["podium"]

X_test = test[features]
y_test = test["podium"]


# ============================================================
# 3. TRAIN XGBOOST
# ============================================================

positive_count = y_train.sum()
negative_count = len(y_train) - positive_count

scale_pos_weight = (
    negative_count / positive_count
)


model = XGBClassifier(

    n_estimators=300,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# 4. GET PODIUM PROBABILITIES
# ============================================================

test["podium_probability"] = (
    model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 5. RACE-LEVEL TOP 3 EVALUATION
# ============================================================

print("\n======================================")
print("RACE-LEVEL PODIUM EVALUATION")
print("======================================")


total_races = 0

driver_hit_count = 0

full_podium_hit_count = 0


race_results = []


for (season, round_number), race in test.groupby(
    ["season", "round"]
):

    # Sort drivers by predicted probability
    predicted = race.sort_values(
        "podium_probability",
        ascending=False
    )

    # Top 3 model predictions
    predicted_top3 = list(
        predicted.head(3)["driver"]
    )

    # Actual podium
    actual = race[
        race["podium"] == 1
    ]

    actual_podium = list(
        actual["driver"]
    )

    # Number of actual podium drivers
    # correctly identified
    hits = len(
        set(predicted_top3)
        & set(actual_podium)
    )

    # Did we identify the entire podium?
    full_hit = (
        set(predicted_top3)
        == set(actual_podium)
    )

    total_races += 1

    driver_hit_count += hits

    if full_hit:
        full_podium_hit_count += 1


    race_results.append({

        "season": season,

        "round": round_number,

        "predicted_top3": ", ".join(
            predicted_top3
        ),

        "actual_podium": ", ".join(
            actual_podium
        ),

        "podium_hits": hits,

        "full_podium_correct": full_hit
    })


# ============================================================
# 6. CALCULATE RESULTS
# ============================================================

average_hits = (
    driver_hit_count / total_races
)

full_podium_accuracy = (
    full_podium_hit_count / total_races
)


print(
    f"\nTest races: {total_races}"
)

print(
    f"Average actual podium drivers "
    f"identified: {average_hits:.2f} / 3"
)

print(
    f"Full podium correctly identified: "
    f"{full_podium_hit_count} / "
    f"{total_races}"
)

print(
    f"Full podium accuracy: "
    f"{full_podium_accuracy * 100:.2f}%"
)


# ============================================================
# 7. SHOW INDIVIDUAL RACES
# ============================================================

results_df = pd.DataFrame(
    race_results
)


print("\n======================================")
print("2025 RACE PREDICTIONS")
print("======================================")


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/race_level_predictions_2025.csv",
    index=False
)


test.to_csv(
    "data/test_predictions_2025.csv",
    index=False
)


print("\nSaved:")

print(
    "data/race_level_predictions_2025.csv"
)

print(
    "data/test_predictions_2025.csv"
)


print("\n======================================")
print("EVALUATION COMPLETE")
print("======================================")