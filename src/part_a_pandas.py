"""Part A, step 6: pandas cross-check of SQL Q2 + exploratory analysis."""
import sqlite3
import pandas as pd

pd.set_option("display.width", 200)
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

# completed orders joined with city and category (same data the SQL used)
done = (orders[orders["status"] == "completed"]
        .merge(customers[["customer_id", "city"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id"))

# ---------- (a) cross-check Q2 with pandas ----------
pivot = pd.pivot_table(done, values="order_amount", index="city",
                       columns="category", aggfunc="sum", margins=True)
print("=== pivot table: revenue by city x category (margins=True adds the totals) ===")
print(pivot.round(2).to_string())

conn = sqlite3.connect("data/instamart.db")
sql_total = pd.read_sql_query(
    "SELECT ROUND(SUM(order_amount), 2) AS t FROM orders WHERE status='completed'", conn)["t"][0]
conn.close()
pivot_total = round(pivot.loc["All", "All"], 2)
print(f"\npivot grand total: {pivot_total:,.2f}   SQL total: {sql_total:,.2f}   equal: {pivot_total == sql_total}")

per_city = done.groupby("city").agg(
    total=("order_amount", "sum"),
    average=("order_amount", "mean"),
    orders=("order_amount", "count"),
).round(2)
print("\n=== per city: sum, mean, count of completed order_amount ===")
print(per_city.to_string())

# ---------- (b) exploratory analysis of completed order_amount ----------
x = done["order_amount"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)
iqr = q3 - q1
low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\n=== order_amount statistics (completed orders) ===")
print(f"mean      {x.mean():.2f}")
print(f"median    {x.median():.2f}")
print(f"skewness  {x.skew():.2f}")
print(f"Q1 {q1:.2f}   Q3 {q3:.2f}   IQR {iqr:.2f}")
print(f"fences: lower {low_fence:.2f}   upper {high_fence:.2f}")
print(f"outliers below lower fence: {int((x < low_fence).sum())}")
print(f"outliers above upper fence: {int((x > high_fence).sum())}")

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[cols].corr()
print("\n=== correlation matrix (completed orders) ===")
print(corr.round(3).to_string())

pairs = corr.where(~(corr.index.to_numpy()[:, None] == corr.columns.to_numpy()[None, :]))
pairs = pairs.stack().abs()
pairs = pairs[[a < b for a, b in pairs.index]]          # each pair only once
a, b = pairs.idxmax()
print(f"\nlargest |correlation|: {a} vs {b} = {corr.loc[a, b]:.3f}")
