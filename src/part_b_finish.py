"""Part B, steps 7-8: class-balance experiment, interpretation, saved model, retention list."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score

from part_b_lapse_data import get_split

X_train, X_test, y_train, y_test = get_split()
saved = joblib.load("data/tuned_models.joblib")
best_lr, best_rf, selected = saved["lr"], saved["rf"], saved["selected"]

# ================= task 7: over-sample the TRAINING set only =================
lapsed_rows = y_train[y_train == 1].index
missing = int((y_train == 0).sum() - (y_train == 1).sum())          # copies needed to balance
extra = pd.Series(lapsed_rows).sample(n=missing, replace=True, random_state=42)
X_bal = pd.concat([X_train, X_train.loc[extra]])
y_bal = pd.concat([y_train, y_train.loc[extra]])
print("=== class-balance experiment ===")
print("training counts after over-sampling:", y_bal.value_counts().sort_index().to_dict(),
      "(reference 570 each)")

balanced_lr = clone(best_lr).fit(X_bal, y_bal)                      # same best C, new data
for label, model in [("logistic regression, original", best_lr),
                     ("logistic regression, balanced ", balanced_lr)]:
    pred = model.predict(X_test)                                    # the ORIGINAL test set
    print(f"{label}: precision {precision_score(y_test, pred, zero_division=0):.3f}  "
          f"recall {recall_score(y_test, pred, zero_division=0):.3f}  "
          f"F1 {f1_score(y_test, pred, zero_division=0):.3f}")

# ================= task 8a: interpret the models =================
names = best_rf.named_steps["pre"].get_feature_names_out()
importance = pd.Series(best_rf.named_steps["model"].feature_importances_, index=names)
print("\n=== five most important features (random forest) ===")
print(importance.sort_values(ascending=False).head(5).round(3).to_string())

coef = pd.Series(best_lr.named_steps["model"].coef_[0],
                 index=best_lr.named_steps["pre"].get_feature_names_out())
print("\n=== logistic regression coefficients for three features (positive = more likely to lapse) ===")
for f in ["num__recency", "num__frequency", "num__cancellation_rate"]:
    print(f"{f:<28} {coef[f]:+.3f}")

# ================= task 8b: save, reload, compare =================
final_model = best_rf if selected == "random forest" else best_lr
import os
os.makedirs("outputs", exist_ok=True)
joblib.dump(final_model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(final_model.predict(X_test), reloaded.predict(X_test))
print(f"\nselected model: {selected}; saved to outputs/lapse_model.joblib")
print("reloaded model gives exactly the same test predictions:", same)

# ================= task 8c: retention list for Premium customers in the test set =================
segments = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
premium_test = [c for c in X_test.index if c in segments.index and bool(segments.loc[c, "is_premium"])]
proba = pd.Series(final_model.predict_proba(X_test)[:, 1], index=X_test.index)
retention = (pd.DataFrame({"lapse_probability": proba.loc[premium_test].round(4),
                           "actually_lapsed": y_test.loc[premium_test]})
             .sort_values("lapse_probability", ascending=False).head(10)
             .rename_axis("customer_id").reset_index())
retention.to_csv("outputs/retention_list.csv", index=False)
print(f"\nPremium customers in the test set: {len(premium_test)}")
print(retention.to_string(index=False))
print(f"\nof these {len(retention)}, {int(retention['actually_lapsed'].sum())} truly lapsed")
print("Note: the retention list is illustrative, not a leakage-free evaluation, because the "
      "Premium flag comes from whole-window RFM that includes September.")
