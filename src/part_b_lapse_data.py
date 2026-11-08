"""Part B, step 4: define the lapse problem, build the features and split the data."""
import pandas as pd
from sklearn.model_selection import train_test_split

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")      # label window: 2025-09-01 up to 2025-09-30

NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct",
           "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
CATEGORICAL = ["favourite_category", "city", "gender"]


def build_dataset():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    # ---- features use ONLY orders dated BEFORE the cutoff (no leakage) ----
    before = orders[orders["order_ts"] < CUTOFF]
    done = before[before["status"] == "completed"]

    feats = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    feats["recency"] = (CUTOFF - feats["last_order"]).dt.days

    # cancellation rate = cancelled orders / all orders with a final status (before the cutoff)
    final = before[before["status"].isin(["completed", "cancelled"])]
    cancel = (final["status"] == "cancelled").groupby(final["customer_id"]).mean()
    feats["cancellation_rate"] = cancel.reindex(feats.index).fillna(0.0)

    # most frequent product category among completed orders (ties -> alphabetical)
    favourite = (done.groupby("customer_id")["category"]
                 .agg(lambda s: s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]))
    feats["favourite_category"] = favourite

    feats = feats.drop(columns="last_order").reset_index().merge(
        customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
    feats["tenure_days"] = (CUTOFF - feats["signup_date"]).dt.days
    feats = feats.drop(columns="signup_date").set_index("customer_id")

    # ---- label: lapsed = 1 if NO completed order from 2025-09-01 to 2025-09-30 ----
    window = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)
                    & (orders["status"] == "completed")]
    active = set(window["customer_id"])
    y = pd.Series([0 if c in active else 1 for c in feats.index],
                  index=feats.index, name="lapsed")

    return feats[NUMERIC + CATEGORICAL], y


def get_split():
    X, y = build_dataset()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


if __name__ == "__main__":
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = get_split()
    print("cohort size:", len(X), "(reference 1,156)")
    print(f"lapsed: {int(y.sum())} ({y.mean() * 100:.2f}%)  (reference 443, 38.32%)")
    print("train rows:", len(X_train), " test rows:", len(X_test), " (reference 924 / 232)")
    print("training lapsed:", int(y_train.sum()), " training not lapsed:", int((y_train == 0).sum()))

    from sklearn.metrics import accuracy_score, f1_score
    always_zero = [0] * len(y_test)
    print(f"\nbaseline 'always not lapsed' on the test set: accuracy "
          f"{accuracy_score(y_test, always_zero) * 100:.1f}%, lapsed-class F1 "
          f"{f1_score(y_test, always_zero, zero_division=0):.2f}  (reference 61.6%, 0)")
    print("\nfeature columns:", list(X.columns))
    print(X.head(3).to_string())
