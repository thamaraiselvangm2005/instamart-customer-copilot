"""Part A, step 2: clean the customers table and save it to data/clean/."""
import os
import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")
print("rows before cleaning:", len(customers))

# (a) CITY: remove spaces at both ends, then Title Case -> 24 spellings become 6
customers["city"] = customers["city"].str.strip().str.title()

# (b) GENDER: every spelling becomes male / female / unknown (blank -> unknown)
def clean_gender(value):
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"
    first_letter = str(value).strip().lower()[0]   # f / m
    return {"f": "female", "m": "male"}.get(first_letter, "unknown")

customers["gender"] = customers["gender"].apply(clean_gender)

# (c) AGE: ages outside 10-100 are not real -> make them missing,
#     then fill ALL missing ages with the MEDIAN AGE OF THAT CUSTOMER'S CITY
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = pd.NA
customers["age"] = customers["age"].astype(float)
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# (d) SIGNUP DATE: text -> real date
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---- save ----
os.makedirs("data/clean", exist_ok=True)
customers.to_csv("data/clean/customers.csv", index=False)

# ---- check against the brief's reference numbers ----
print("\ncities:\n", customers["city"].value_counts().to_string())
print("\ngenders:\n", customers["gender"].value_counts().to_string())
print("\nmissing ages:", int(customers["age"].isna().sum()))
print("age range:", customers["age"].min(), "to", customers["age"].max())
