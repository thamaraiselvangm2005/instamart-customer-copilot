"""Part A, step 5: run the seven queries in sql/queries.sql against data/instamart.db."""
import re
import sqlite3
import pandas as pd

# ---- read the .sql file and cut it into 7 queries (each starts with "-- Q<number>") ----
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"(?m)^-- (Q\d+):[^\n]*\n", text)   # ['', 'Q1', body, 'Q2', body, ...]
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}

conn = sqlite3.connect("data/instamart.db")
result = {name: pd.read_sql_query(sql, conn) for name, sql in queries.items()}
conn.close()

def show(name, title):
    print(f"\n=== {name}: {title} ===")

q1, q2, q3, q4, q5, q6, q7 = (result[f"Q{i}"] for i in range(1, 8))

show("Q1", "customers with big completed orders")
print("customers:", len(q1), "(reference 30)")

show("Q2", "revenue by city and category")
print("rows:", len(q2), "(reference 30)")
print(f"sum of revenue: INR {q2['revenue'].sum():,.2f} (reference 2,998,668.80)")
top = q2.iloc[0]
print(f"largest cell: {top['city']} / {top['category']} INR {top['revenue']:,.2f} (reference Bengaluru / Electronics 233,909.45)")

show("Q3", "products above average price")
print("products:", len(q3), "(reference 11)")

show("Q4", "customers with no completed order")
print("customers:", len(q4), "(reference 25)")

show("Q5", "customers above average spend (CTE)")
print("customers:", len(q5), "(reference 550)")
print("average spend: INR", q5["avg_spend"].iloc[0], "(reference 2552.06)")

show("Q6", "rank inside each city")
print("top customer per city (row_num = 1):")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("rows where RANK differs from ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()), "(reference 8)")

show("Q7", "daily revenue with running total")
print("daily rows:", len(q7), "(reference 183)")
print(f"last cumulative value: INR {q7['cumulative_revenue'].iloc[-1]:,.2f} (reference 2,998,668.80)")
