"""
Train the NBA combine -> career outcome models.

Two things are trained on data/processed/player_data.csv (the combine ∩
career intersection built by preprocess_data.py):

1. A classifier predicting the outcome `category` (Bust / Role Player /
   Starter / Star) directly from combine measurables -- the actual
   star/bust predictor. Three model families (Random Forest, Logistic
   Regression, Gradient Boosting) are each hyperparameter-tuned with
   GridSearchCV, so a weak result can't be explained away as "the wrong
   model" or "bad hyperparameters" -- the winner (by cross-validated
   accuracy) is deployed, and every candidate's numbers are kept in
   metrics.json so the report can show the full comparison.
2. RandomForestRegressors predicting career PPG/RPG/APG from the same
   measurables, used only to find a "comparable career" for the
   frontend (nearest-neighbor by predicted stat line).
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/player_data.csv"

FEATURE_COLUMNS = [
    "HGT", "WGT", "BMI", "WNGSPN", "STNDRCH", "BAR",
    "STNDVERT", "LANE", "SPRINT",
]
CATEGORY_ORDER = ["Bust", "Role Player", "Starter", "Star"]

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Three different model families, each with its own small hyperparameter
# grid. Logistic Regression is wrapped in a scaling Pipeline so the saved
# artifact behaves like any other sklearn estimator (predict / predict_proba
# / classes_) and the API doesn't need to know which model type won.
MODEL_CANDIDATES = {
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8, None],
            "min_samples_leaf": [1, 3, 5, 10],
        },
    ),
    "Logistic Regression": (
        Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.1],
        },
    ),
}


def tune_candidates(X_train, y_train):
    """Grid-search every candidate model family and return their fitted
    GridSearchCV objects, keyed by model name."""
    results = {}
    for name, (estimator, param_grid) in MODEL_CANDIDATES.items():
        search = GridSearchCV(estimator, param_grid, cv=CV, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)
        results[name] = search
        print(f"  {name:20s} best CV accuracy: {search.best_score_:.3f}  params: {search.best_params_}")
    return results


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

    # --- Tune each model family, select the winner by CV accuracy (not
    # test accuracy, so the model comparison doesn't leak test-set info
    # into model selection) ---
    print("Tuning candidate models (5-fold CV, grid search):")
    searches = tune_candidates(X_train, y_train)

    winner_name = max(searches, key=lambda n: searches[n].best_score_)
    winner_search = searches[winner_name]
    clf = winner_search.best_estimator_
    # Pull the winner's own CV mean/std (from its best hyperparameter combo)
    # instead of re-running cross_val_score separately -- that would CV over
    # the full X/y, letting held-out test rows leak into training folds.
    winner_cv_mean = winner_search.cv_results_["mean_test_score"][winner_search.best_index_]
    winner_cv_std = winner_search.cv_results_["std_test_score"][winner_search.best_index_]
    print(f"\nWinner: {winner_name} (best CV accuracy {winner_search.best_score_:.3f})")

    model_comparison = [
        {
            "name": name,
            "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                             for k, v in search.best_params_.items()},
            "cv_accuracy": round(float(search.best_score_), 4),
            "test_accuracy": round(float(accuracy_score(y_test, search.best_estimator_.predict(X_test))), 4),
        }
        for name, search in searches.items()
    ]
    model_comparison.sort(key=lambda m: m["cv_accuracy"], reverse=True)

    test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)

    report = classification_report(y_test, test_pred, labels=CATEGORY_ORDER, output_dict=True)
    cm = confusion_matrix(y_test, test_pred, labels=CATEGORY_ORDER)

    # Permutation importance works for any model type (unlike
    # feature_importances_, which only tree ensembles expose), so it stays
    # valid no matter which candidate wins.
    perm = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE)
    importances = dict(zip(FEATURE_COLUMNS, perm.importances_mean.round(4)))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    metrics = {
        "winning_model": winner_name,
        "test_accuracy": round(float(test_accuracy), 4),
        "baseline_accuracy": round(float(baseline_accuracy), 4),
        "cv_accuracy_mean": round(float(winner_cv_mean), 4),
        "cv_accuracy_std": round(float(winner_cv_std), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_order": CATEGORY_ORDER,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": importances,
        "model_comparison": model_comparison,
    }

    print(f"\nTest accuracy ({winner_name}): {test_accuracy:.3f}")
    print(f"Baseline accuracy:  {baseline_accuracy:.3f}  (always predicts majority class)")
    print(f"5-fold CV accuracy: {winner_cv_mean:.3f} +/- {winner_cv_std:.3f}")
    print("\nPermutation importances:")
    for feat, imp in importances.items():
        print(f"  {feat:10s} {imp:.4f}")

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
