"""Part B, steps 5-6: preprocess without leakage, tune two models, evaluate on the test set."""
import os
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from part_b_lapse_data import CATEGORICAL, NUMERIC, build_dataset, get_split


def make_pipeline(model):
    """One-hot (k-1 columns) + scaling are INSIDE the pipeline, so they are fitted on training rows only."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


if __name__ == "__main__":
    X_all, y_all = build_dataset()
    X_train, X_test, y_train, y_test = get_split()

    # ---------- task 5: no leakage checks ----------
    print("=== preprocessing checks ===")
    print("missing values in any feature column:", int(X_all.isna().sum().sum()), "(must be 0)")
    probe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    scaler = probe.named_steps["pre"].named_transformers_["num"]
    learned = scaler.mean_[NUMERIC.index("frequency")]
    print(f"frequency mean learned by the scaler (training rows only): {learned:.3f}")
    print(f"frequency mean of the full cohort:                        {X_all['frequency'].mean():.3f}")
    print("They differ because the scaler saw only the 924 training rows (80% of customers), "
          "while the full-cohort mean also includes the 232 test rows the scaler must never see.")

    # ---------- task 6: tune both models (F1 of the lapsed class, 5-fold CV) ----------
    print("\n=== tuning ===")
    lr_grid = GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000)),
                           {"model__C": [0.01, 0.1, 1, 10, 100]},
                           cv=5, scoring="f1").fit(X_train, y_train)
    rf_grid = GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
                           {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
                           cv=5, scoring="f1").fit(X_train, y_train)
    print("logistic regression best:", lr_grid.best_params_, f"CV F1 = {lr_grid.best_score_:.3f}")
    print("random forest best:      ", rf_grid.best_params_, f"CV F1 = {rf_grid.best_score_:.3f}")

    # ---------- evaluation on the untouched test set (lapsed class = positive) ----------
    print("\n=== test-set results (lapsed class) ===")
    scores = {}
    for name, grid in [("logistic regression", lr_grid), ("random forest", rf_grid)]:
        pred = grid.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        p = precision_score(y_test, pred, zero_division=0)
        r = recall_score(y_test, pred, zero_division=0)
        f = f1_score(y_test, pred, zero_division=0)
        scores[name] = f
        print(f"\n{name}")
        print("confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())
        print(f"precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}")

    # choose with the cross-validation score, so the test set is not used to pick the winner
    cv_scores = {"logistic regression": lr_grid.best_score_, "random forest": rf_grid.best_score_}
    selected = max(cv_scores, key=cv_scores.get)
    print("\nselected model (higher cross-validated F1):", selected,
          f"-> test F1 {scores[selected]:.3f} (must be at least 0.55)")

    os.makedirs("data", exist_ok=True)
    joblib.dump({"lr": lr_grid.best_estimator_, "rf": rf_grid.best_estimator_,
                 "selected": selected}, "data/tuned_models.joblib")
    print("saved the tuned models to data/tuned_models.joblib (used by the next script)")
