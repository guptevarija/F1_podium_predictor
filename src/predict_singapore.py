import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ============================================================
# 2026 SINGAPORE GP
# PRE-QUALIFYING PODIUM PREDICTION
# ============================================================

TRAIN_FILE = "data/f1_model_dataset_v3.csv"
PREDICTION_FILE = "data/f1_singapore_2026_features.csv"

OUTPUT_FILE = "data/singapore_2026_prediction.csv"


print("=" * 70)
print("2026 SINGAPORE GP - PRE-QUALIFYING PREDICTION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
prediction = pd.read_csv(PREDICTION_FILE)

print("\nTraining rows:", len(train))
print("Singapore drivers:", len(prediction))


# ============================================================
# 2. FEATURES
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
# 3. CHECK FEATURES
# ============================================================

print("\nChecking training features...")

missing_train = [
    f for f in features
    if f not in train.columns
]

missing_prediction = [
    f for f in features
    if f not in prediction.columns
]

if missing_train:
    print("\nMissing from training dataset:")
    print(missing_train)
    raise SystemExit

if missing_prediction:
    print("\nMissing from prediction dataset:")
    print(missing_prediction)
    raise SystemExit


# ============================================================
# 4. TARGET
# ============================================================

if "podium" not in train.columns:

    print("\nERROR: 'podium' target is missing.")
    raise SystemExit


X_train = train[features].copy()
y_train = train["podium"].copy()

X_predict = prediction[features].copy()


# ============================================================
# 5. REMOVE INVALID TARGET ROWS
# ============================================================

valid_target = y_train.notna()

X_train = X_train.loc[valid_target]
y_train = y_train.loc[valid_target]


# ============================================================
# 6. IMPUTE MISSING VALUES
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_predict_imputed = imputer.transform(
    X_predict
)


# ============================================================
# 7. CLASS BALANCE
# ============================================================

positive = (y_train == 1).sum()
negative = (y_train == 0).sum()

scale_pos_weight = negative / positive

print("\nPodium observations:", positive)
print("Non-podium observations:", negative)
print(
    "Scale positive weight:",
    round(scale_pos_weight, 2)
)


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")


model = XGBClassifier(

    n_estimators=400,

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
    X_train_imputed,
    y_train
)


print("Model trained successfully.")


# ============================================================
# 9. PREDICT PODIUM PROBABILITY
# ============================================================

probabilities = model.predict_proba(
    X_predict_imputed
)[:, 1]


prediction["podium_probability"] = probabilities


# Convert probability to percentage
prediction["podium_probability_pct"] = (
    prediction["podium_probability"] * 100
)


# ============================================================
# 10. RANK DRIVERS
# ============================================================

prediction = prediction.sort_values(
    "podium_probability",
    ascending=False
).reset_index(drop=True)


prediction["predicted_rank"] = (
    prediction.index + 1
)


# ============================================================
# 11. SAVE
# ============================================================

prediction.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 12. DISPLAY FULL RANKING
# ============================================================

print("\n" + "=" * 70)
print("SINGAPORE GP DRIVER RANKING")
print("=" * 70)

print(
    prediction[
        [
            "predicted_rank",
            "driver",
            "constructor",
            "podium_probability_pct"
        ]
    ].to_string(
        index=False,
        formatters={
            "podium_probability_pct":
            lambda x: f"{x:.2f}%"
        }
    )
)


# ============================================================
# 13. PREDICTED PODIUM
# ============================================================

top_3 = prediction.head(3)


print("\n" + "=" * 70)
print("🏆 PREDICTED 2026 SINGAPORE PODIUM")
print("=" * 70)

for i, (_, row) in enumerate(
    top_3.iterrows(),
    start=1
):

    print(
        f"{i}. {row['driver']} "
        f"({row['constructor']}) "
        f"- {row['podium_probability_pct']:.2f}%"
    )


# ============================================================
# 14. SAVE FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    "data/singapore_2026_feature_importance.csv",
    index=False
)


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nPrediction:"
)

print(
    OUTPUT_FILE
)

print(
    "\nFeature importance:"
)

print(
    "data/singapore_2026_feature_importance.csv"
)

print("\n" + "=" * 70)
print("PRE-QUALIFYING PREDICTION COMPLETE")
print("=" * 70)