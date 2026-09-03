"""
Train the NBA combine -> career outcome models.

Two things are trained on data/processed/player_data.csv (the combine ∩
career intersection built by preprocess_data.py):

1. A RandomForestClassifier predicting the outcome `category`
   (Bust / Role Player / Starter / Star) directly from combine
   measurables. This is the actual star/bust predictor -- and its
   accuracy against a majority-class baseline is the headline number
   for "do combine scores mean anything".
2. RandomForestRegressors predicting career PPG/RPG/APG from the same
   measurables, used only to find a "comparable career" for the
   frontend (nearest-neighbor by predicted stat line).
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

DATA_PATH = "data/processed/player_data.csv"

FEATURE_COLUMNS = [
    "HGT", "WGT", "BMI", "WNGSPN", "STNDRCH", "BAR",
    "STNDVERT", "LANE", "SPRINT",
]
CATEGORY_ORDER = ["Bust", "Role Player", "Starter", "Star"]

RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Classifier: predicted category ---
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    cv_scores = cross_val_score(clf, X, y, cv=5)

    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

    report = classification_report(y_test, test_pred, labels=CATEGORY_ORDER, output_dict=True)
    cm = confusion_matrix(y_test, test_pred, labels=CATEGORY_ORDER)

    importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.round(4)))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    metrics = {
        "test_accuracy": round(float(test_accuracy), 4),
        "baseline_accuracy": round(float(baseline_accuracy), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_order": CATEGORY_ORDER,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": importances,
    }

    print(f"Test accuracy:      {test_accuracy:.3f}")
    print(f"Baseline accuracy:  {baseline_accuracy:.3f}  (always predicts majority class)")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print("\nFeature importances:")
    for feat, imp in importances.items():
        print(f"  {feat:10s} {imp:.3f}")

    joblib.dump(clf, "data/processed/category_model.pkl")
    with open("data/processed/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Regressors: predicted career PPG/RPG/APG, for the "comparable
    # player" lookup only. Fit on the full matched dataset since they
    # aren't the thing being evaluated for accuracy. ---
    reg_kwargs = dict(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=RANDOM_STATE)
    ppg_reg = RandomForestRegressor(**reg_kwargs).fit(X, df["PPG"])
    rpg_reg = RandomForestRegressor(**reg_kwargs).fit(X, df["RPG"])
    apg_reg = RandomForestRegressor(**reg_kwargs).fit(X, df["APG"])

    joblib.dump(ppg_reg, "data/processed/ppg_reg.pkl")
    joblib.dump(rpg_reg, "data/processed/rpg_reg.pkl")
    joblib.dump(apg_reg, "data/processed/apg_reg.pkl")

    # Reference stat lines + names for the nearest-comparable-player lookup.
    career_matrix = df[["PPG", "RPG", "APG"]].to_numpy()
    joblib.dump(career_matrix, "data/processed/career_matrix.pkl")
    joblib.dump(df["PLAYER"].tolist(), "data/processed/career_players.pkl")

    # Median feature values (for imputing missing frontend inputs) and
    # the feature list, so the API and training stay in sync.
    joblib.dump(
        {"feature_columns": FEATURE_COLUMNS, "medians": X.median().to_dict(), "class_order": CATEGORY_ORDER},
        "data/processed/feature_meta.pkl",
    )

    print("\nSaved: category_model.pkl, ppg_reg.pkl, rpg_reg.pkl, apg_reg.pkl, "
          "career_matrix.pkl, career_players.pkl, feature_meta.pkl, metrics.json")


if __name__ == "__main__":
    main()
