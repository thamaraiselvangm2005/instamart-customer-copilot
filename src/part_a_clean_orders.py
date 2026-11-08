"""Part A, step 3: clean the orders table and save it to data/clean/."""
import os
import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")
print("rows at start:", len(orders))

# RULE 1: order_ts text -> real date and time
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# RULE 2: remove rows that are EXACT copies of another row
before = len(orders)
orders = orders.drop_duplicates()
print(f"rule 2: exact duplicates removed: {before - len(orders)}")

# RULE 3: some orders appear twice with different statuses (an early
# 'pending' row and a later final row). Keep only the LATEST row per order_id.
before = len(orders)
orders = orders.sort_values("order_ts").drop_duplicates("order_id", keep="last")
print(f"rule 3: older rows of repeated order_id removed: {before - len(orders)}")

# RULE 4: order_amount -> number. Unreadable or blank amounts become missing,
# then are rebuilt from the product price:
#     round(price x quantity x (1 - discount/100), 2)
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "category", "price_inr"]],
                      on="product_id", how="left")
missing_amount = orders["order_amount"].isna()
rebuilt = (orders["price_inr"] * orders["quantity"]
           * (1 - orders["discount_pct"] / 100)).round(2)
orders.loc[missing_amount, "order_amount"] = rebuilt[missing_amount]
print(f"rule 4: amounts rebuilt from product price: {int(missing_amount.sum())}")

# RULE 5: delivery_minutes. A negative time is impossible -> make it missing.
#   completed orders: fill with the median time of that product category
#                     (median taken among completed orders only)
#   cancelled orders: LEAVE MISSING (they were never delivered)
negative = orders["delivery_minutes"] < 0
orders.loc[negative, "delivery_minutes"] = pd.NA
orders["delivery_minutes"] = orders["delivery_minutes"].astype(float)
print(f"rule 5: negative delivery times set to missing: {int(negative.sum())}")

completed = orders["status"] == "completed"
category_median = (orders["delivery_minutes"].where(completed)
                   .groupby(orders["category"]).transform("median"))
fill_rows = completed & orders["delivery_minutes"].isna()
orders.loc[fill_rows, "delivery_minutes"] = category_median[fill_rows]
print(f"rule 5: completed delivery times filled by category median: {int(fill_rows.sum())}")

# ---- save (order_ts stored as text YYYY-MM-DD HH:MM:SS) ----
clean = orders[["order_id", "customer_id", "product_id", "quantity", "order_ts",
                "status", "discount_pct", "order_amount", "delivery_minutes"]].copy()
clean = clean.sort_values("order_id").reset_index(drop=True)
clean["order_ts"] = clean["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
os.makedirs("data/clean", exist_ok=True)
clean.to_csv("data/clean/orders.csv", index=False)

# ---- check against the brief's reference numbers ----
done = clean["status"] == "completed"
print("\nrows after cleaning:", len(clean))
print(clean["status"].value_counts().to_string())
print("missing order_amount:", int(clean["order_amount"].isna().sum()))
print("missing delivery (completed):", int(clean.loc[done, "delivery_minutes"].isna().sum()))
print("missing delivery (cancelled):",
      int(clean.loc[clean["status"] == "cancelled", "delivery_minutes"].isna().sum()))
print("sum of order_amount, completed: INR", f"{clean.loc[done, 'order_amount'].sum():,.2f}")
