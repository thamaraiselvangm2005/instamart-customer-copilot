"""Part A, step 4: store the clean tables (CSV + SQLite) and validate them.

Run AFTER part_a_clean_customers.py and part_a_clean_orders.py.
"""
import os
import sqlite3
import pandas as pd

# ---- 1. products need no cleaning: copy them into data/clean/ ----
products = pd.read_csv("data/raw/products.csv")
products.to_csv("data/clean/products.csv", index=False)

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

# ---- 2. write the three tables into the SQLite database ----
if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")          # start fresh every time
conn = sqlite3.connect("data/instamart.db")
customers.to_sql("customers", conn, index=False)
products.to_sql("products", conn, index=False)
orders.to_sql("orders", conn, index=False)   # order_ts is already text YYYY-MM-DD HH:MM:SS

# ---- 3. validation: compare with the reference numbers from the brief ----
# (we read the numbers back FROM THE DATABASE, so we test what was stored)
db_customers = pd.read_sql_query("SELECT * FROM customers", conn)
db_orders = pd.read_sql_query("SELECT * FROM orders", conn)
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"]
conn.close()

done = db_orders["status"] == "completed"
cancelled = db_orders["status"] == "cancelled"
city = db_customers["city"].value_counts()
gender = db_customers["gender"].value_counts()

checks = {
    "database has the 3 tables": set(tables) == {"customers", "products", "orders"},
    "exactly 6 cities": db_customers["city"].nunique() == 6,
    "city counts Pune 227, Delhi 210, Bengaluru 197": (
        city["Pune"], city["Delhi"], city["Bengaluru"]) == (227, 210, 197),
    "genders female 610, male 560, unknown 30": (
        gender["female"], gender["male"], gender["unknown"]) == (610, 560, 30),
    "0 missing ages": db_customers["age"].isna().sum() == 0,
    "every age between 18 and 60": db_customers["age"].between(18, 60).all(),
    "12,309 order rows, one per order_id": (
        len(db_orders) == 12309 and db_orders["order_id"].is_unique),
    "11,110 completed and 1,199 cancelled": (done.sum(), cancelled.sum()) == (11110, 1199),
    "no pending rows": (db_orders["status"] == "pending").sum() == 0,
    "0 missing order_amount": db_orders["order_amount"].isna().sum() == 0,
    "0 missing delivery among completed": db_orders.loc[done, "delivery_minutes"].isna().sum() == 0,
    "1,199 missing delivery among cancelled": db_orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199,
    "completed revenue is INR 2,998,668.80": round(db_orders.loc[done, "order_amount"].sum(), 2) == 2998668.80,
}

all_ok = True
for label, passed in checks.items():
    print(("PASS  " if passed else "FAIL  ") + label)
    all_ok = all_ok and bool(passed)

assert all_ok, "At least one checkpoint failed - see the FAIL lines above"
print("ALL CHECKPOINTS PASSED")
