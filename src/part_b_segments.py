"""Part B, step 3: final K-means (K=4), name the segments, run PCA, save the segment file."""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from part_b_rfm import build_rfm

FINAL_K = 4   # chosen from {3, 4, 5}: best silhouette (0.526), clear elbow, 4 groups the business can act on

rfm = build_rfm()
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rfm["segment_id"] = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit_predict(X)

# ---- (a) profile: size and mean R, F, M of every segment ----
profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(2)

# ---- (b) business names. Cluster numbers are only labels with NO order,
# so the names are decided from the numbers, never from the cluster number. ----
names, actions = {}, {}
premium_id = profile["mean_monetary"].idxmax()                  # highest mean monetary = Premium
names[premium_id] = "Premium Big Spenders"
actions[premium_id] = "VIP treatment: early access, personal offers, priority support"

rest = profile.drop(index=premium_id)
dormant_id = rest["mean_recency"].idxmax()                      # longest time since last order
names[dormant_id] = "Dormant Customers"
actions[dormant_id] = "Win-back campaign: reminder message plus a comeback discount"

rest = rest.drop(index=dormant_id)
loyal_id = rest["mean_frequency"].idxmax()                      # orders most often
names[loyal_id] = "Loyal Regulars"
actions[loyal_id] = "Reward loyalty: points, subscriptions, bundles to raise basket size"

casual_id = rest.drop(index=loyal_id).index[0]                  # the one left
names[casual_id] = "Casual Shoppers"
actions[casual_id] = "Nudge to order more often: free-delivery threshold, category offers"

profile["segment_name"] = pd.Series(names)
profile["recommended_action"] = pd.Series(actions)
profile["is_premium"] = profile.index == premium_id
print("=== segment profile (K = 4) ===")
print(profile.to_string())
print("\nReminder: segment_id numbers are only labels; they have no order or meaning.")

# ---- (c) PCA on the scaled RFM matrix ----
pca = PCA(n_components=2).fit(X)
ratio = pca.explained_variance_ratio_
print(f"\nPCA explained variance: {ratio[0]:.3f} and {ratio[1]:.3f}, total {ratio.sum():.3f}"
      "  (reference about 0.634, 0.218, total 0.852)")

# ---- save outputs/customer_segments.csv ----
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
os.makedirs("outputs", exist_ok=True)
rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("\nsaved outputs/customer_segments.csv with", len(rfm), "rows;",
      "Premium customers:", int(rfm["is_premium"].sum()))
