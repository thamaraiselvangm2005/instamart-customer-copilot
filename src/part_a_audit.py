"""Part A, step 1: audit the three raw tables (read-only, changes nothing)."""
import pandas as pd

RAW = "data/raw/"

orders = pd.read_csv(RAW + "orders.csv")
customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")

# ---- 1. For each table: shape, data types, missing values per column ----
for name, df in [("orders", orders), ("customers", customers), ("products", products)]:
    print(f"\n=== {name}: shape = {df.shape} ===")
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    })
    print(report.to_string())

# ---- 2. The specific audit figures ----
amount_as_number = pd.to_numeric(orders["order_amount"], errors="coerce")
# "cannot be converted" counts blank entries too (53 blank + 67 text like "N/A" = 120)
bad_amount = amount_as_number.isna()
delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

age = pd.to_numeric(customers["age"], errors="coerce")
gender_raw = customers["gender"].dropna().astype(str)
gender_nonblank = gender_raw[gender_raw.str.strip() != ""]

figures = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "rows with status pending": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "order_amount not convertible to a number": int(bad_amount.sum()),
    "negative delivery_minutes": int((delivery < 0).sum()),
    "blank delivery_minutes": int(delivery.isna().sum()),
    "customers: blank age": int(age.isna().sum()),
    "customers: age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "customers: distinct raw city spellings": int(customers["city"].nunique()),
    "customers: distinct non-blank gender spellings": int(gender_nonblank.nunique()),
    "customers: blank gender": int(len(customers) - len(gender_nonblank)),
}
reference = [13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30]

print("\n=== AUDIT FIGURES (yours vs brief reference) ===")
for (label, value), ref in zip(figures.items(), reference):
    mark = "OK " if value == ref else "DIFF"
    print(f"{mark} {label:<48} {value:>7,}   (reference {ref:,})")
