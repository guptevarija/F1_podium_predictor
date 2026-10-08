import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD TRAIN / VALIDATION / TEST DATA
# ============================================================

train = pd.read_csv("data/train.csv")
validation = pd.read_csv("data/validation.csv")
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


target = "podium"


X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]


print("\n======================================")
print("F1 PODIUM PREDICTION")
print("======================================")

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))


# ============================================================
# 3. HELPER FUNCTION FOR EVALUATION
# ============================================================

def evaluate_model(name, model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    print(f"\n{name}")

    print("-" * 40)

    print(
        f"Precision : {precision:.3f}"
    )

    print(
        f"Recall    : {recall:.3f}"
    )

    print(
        f"F1 Score  : {f1:.3f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.3f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.3f}"
    )

    return probabilities


# ============================================================
# 4. SIMPLE BASELINE
# ============================================================

# Our first baseline:
#
# Predict that drivers with qualifying position <= 3
# will podium.
#
# This gives us something very simple to beat.


baseline_predictions = (
    X_validation["qualifying_position"] <= 3
).astype(int)


baseline_precision = precision_score(
    y_validation,
    baseline_predictions,
    zero_division=0
)

baseline_recall = recall_score(
    y_validation,
    baseline_predictions,
    zero_division=0
)

baseline_f1 = f1_score(
    y_validation,
    baseline_predictions,
    zero_division=0
)


print("\n======================================")
print("BASELINE — TOP 3 QUALIFYING")
print("======================================")

print(
    f"Precision : {baseline_precision:.3f}"
)

print(
    f"Recall    : {baseline_recall:.3f}"
)

print(
    f"F1 Score  : {baseline_f1:.3f}"
)


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

# Missing values are filled using training-set medians.

logistic_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)


logistic_pipeline.fit(
    X_train,
    y_train
)


print("\n======================================")
print("LOGISTIC REGRESSION")
print("======================================")


logistic_validation_probs = evaluate_model(
    "Validation Results",
    logistic_pipeline,
    X_validation,
    y_validation
)


# ============================================================
# 6. XGBOOST
# ============================================================

# Calculate class imbalance.

positive_count = y_train.sum()

negative_count = len(y_train) - positive_count

scale_pos_weight = (
    negative_count / positive_count
)


xgb_model = XGBClassifier(

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


# XGBoost can handle missing values itself.

xgb_model.fit(
    X_train,
    y_train
)


print("\n======================================")
print("XGBOOST")
print("======================================")


xgb_validation_probs = evaluate_model(
    "Validation Results",
    xgb_model,
    X_validation,
    y_validation
)


# ============================================================
# 7. FINAL TEST — 2025
# ============================================================

print("\n======================================")
print("FINAL TEST — 2025")
print("======================================")


print("\nXGBoost:")

xgb_test_probs = evaluate_model(
    "2025 Test Results",
    xgb_model,
    X_test,
    y_test
)


print("\nLogistic Regression:")

logistic_test_probs = evaluate_model(
    "2025 Test Results",
    logistic_pipeline,
    X_test,
    y_test
)


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": features,

    "importance": (
        xgb_model.feature_importances_
    )

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n======================================")
print("XGBOOST FEATURE IMPORTANCE")
print("======================================")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(
    "data/xgboost_feature_importance.csv",
    index=False
)


print(
    "\nFeature importance saved to:"
)

print(
    "data/xgboost_feature_importance.csv"
)


print("\n======================================")
print("MODEL TRAINING COMPLETE")
print("======================================")