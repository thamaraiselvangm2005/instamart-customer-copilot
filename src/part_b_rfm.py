"""Part B, step 1: build the RFM table (recency, frequency, monetary) per customer."""
import pandas as pd

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")


def build_rfm(path="data/clean/orders.csv"):
    orders = pd.read_csv(path, parse_dates=["order_ts"])
    done = orders[orders["status"] == "completed"]          # only completed orders count

    rfm = done.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),                    # distinct completed orders
        monetary=("order_amount", "sum"),                     # total money spent
    )
    # recency = whole days between the snapshot and the most recent order
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
    return rfm


if __name__ == "__main__":
    rfm = build_rfm()
    print("customers with at least one completed order:", len(rfm), "(reference 1,175)")
    print("recency  median:", rfm["recency"].median(), "(ref 19)",
          " min:", rfm["recency"].min(), "(ref 0)",
          " max:", rfm["recency"].max(), "(ref 180)")
    print("frequency mean:", round(rfm["frequency"].mean(), 2), "(ref 9.46)",
          " max:", rfm["frequency"].max(), "(ref 33)")
    print("monetary  mean:", round(rfm["monetary"].mean(), 2), "(ref 2,552.06)",
          " max:", round(rfm["monetary"].max(), 2), "(ref 14,766.55)")
    print("\nfirst 5 rows:")
    print(rfm.head().to_string(index=False))
