# Instamart Customer Intelligence and Support Copilot

## Overview

This project works on the data of an online grocery delivery business (Instamart-style). It has four parts:

- **Part A:** audit and clean the raw data, store it in SQLite, and answer business questions with SQL and pandas.
- **Part B:** customer segments and a lapse-prediction model.
- **Part C:** support-ticket triage.
- **Part D:** policy question answering and a refund agent.

All language-model calls use the supplied offline file `mock_llm.py`, so no API key, account or internet connection is needed. All amounts are in INR.

## Reproduce

Run these commands in order from the repository folder (Python 3.11 or newer).

1. `python -m venv .venv`
2. Activate it: `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python src/part_a_audit.py`
6. `python src/part_a_clean_customers.py`
7. `python src/part_a_clean_orders.py`
8. `python src/part_a_store_validate.py`
9. `python src/part_a_sql.py`
10. `python src/part_a_pandas.py`

Parts B, C and D commands will be added here as those parts are finished.

## Part A

### Audit of the raw data

| Check | Value |
| --- | --- |
| Raw orders rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| order_amount not convertible to a number | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Customers: blank age | 61 |
| Customers: age outside 10-100 | 6 |
| Customers: distinct raw city spellings | 24 |
| Customers: distinct non-blank gender spellings | 10 |
| Customers: blank gender | 30 |

Shapes: orders (13,084, 9), customers (1,200, 5), products (40, 4). Missing values: orders has 53 blank order_amount (0.41%) and 2,541 blank delivery_minutes (19.42%); customers has 61 blank age (5.08%) and 30 blank gender (2.50%); products has none.

### Cleaning rules

**Customers**

| Rule | Values changed |
| --- | --- |
| City: trim spaces and apply title case (24 spellings become 6 cities) | 476 |
| Gender: map every spelling to male, female or unknown (blank becomes unknown) | 950 (including 30 blank) |
| Age: treat ages outside 10-100 as missing, then fill all missing ages with the median age of the customer's city | 67 filled (61 blank + 6 out of range) |
| signup_date converted to a datetime | all 1,200 |

**Orders** (applied in this order)

| Rule | Rows or values changed |
| --- | --- |
| 1. Convert order_ts to a datetime | all 13,084 rows |
| 2. Remove exact duplicate rows | 119 removed |
| 3. Keep only the record with the latest order_ts for each order_id | 656 older rows removed |
| 4. Convert order_amount to numeric; rebuild missing amounts as round(price_inr x quantity x (1 - discount_pct / 100), 2) | 112 amounts rebuilt |
| 5a. Set negative delivery_minutes to missing | 105 set to missing |
| 5b. Completed orders: fill missing delivery_minutes with the median of the product category among completed orders | 774 filled |
| 5c. Cancelled orders: leave delivery_minutes missing (a cancelled order was never delivered) | 1,199 left missing |

Result: 12,309 orders (11,110 completed and 1,199 cancelled), no pending rows, 0 missing order_amount, and completed revenue of INR 2,998,668.80.

### Storage and validation

The clean tables are saved as `data/clean/customers.csv`, `data/clean/products.csv` and `data/clean/orders.csv`, and in the SQLite database `data/instamart.db` (tables customers, products, orders; order_ts stored as text YYYY-MM-DD HH:MM:SS). `src/part_a_store_validate.py` checks 13 reference points and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis

The seven queries are in `sql/queries.sql` and are run from Python by `src/part_a_sql.py`.

| Query | Result |
| --- | --- |
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; revenues add up to INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 common table expressions | 550 customers above the average total of INR 2,552.06 |
| Q6 ranking window functions | Top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows where RANK differs from ROW_NUMBER |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

RANK and ROW_NUMBER differ here because customers with the same spend in a city share the same RANK and the next RANK skips ahead, while ROW_NUMBER always counts 1, 2, 3 without ties.

### pandas cross-check

The pivot table (`pd.pivot_table` with `margins=True`) gives a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Statistics of completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30, 0 outliers below and 897 above.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
| --- | --- | --- | --- | --- |
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

### Exploratory analysis

An online grocery business shared data on 12,309 orders from 1,200 customers in 6 cities, covering 40 products in 5 categories. After cleaning the data, 11,110 orders were completed and 1,199 were cancelled, so we studied only the completed ones.

Order amounts are right-skewed (skewness 4.61). The average order is INR 269.91, but the middle order (the median) is only INR 135.00. This happens because most orders are small, while a few very big orders pull the average up. The middle half of all orders falls between INR 64.80 and INR 297.00.

Using the 1.5 × IQR rule, any order above INR 645.30 is an outlier, which means it is much bigger than normal. There are 897 such orders, and none fall below the lower fence. They are probably real purchases, like costly products or lots of items, and not mistakes.

We also checked whether order amount, discount, quantity and delivery time are connected. All the links were very weak. The strongest one, between order amount and discount, was only about -0.095, which is almost nothing. A correlation this small does not establish causation.

## Part B

To be completed.

## Part C

To be completed.

## Part D

To be completed.

## Limitations and Next Steps

To be completed.
