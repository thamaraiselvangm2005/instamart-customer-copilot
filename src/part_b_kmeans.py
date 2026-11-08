"""Part B, step 2: K-means for K = 2..8 on the standardised RFM table."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from part_b_rfm import build_rfm

rfm = build_rfm()
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

reference = {2: (1820.8, 0.465), 3: (1268.2, 0.445), 4: (743.1, 0.526),
             5: (617.4, 0.481), 6: (503.2, 0.482), 7: (408.5, 0.416), 8: (366.5, 0.412)}

rows = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k,
                 "inertia": round(model.inertia_, 1),
                 "silhouette": round(silhouette_score(X, model.labels_), 3),
                 "ref_inertia": reference[k][0],
                 "ref_silhouette": reference[k][1]})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
print("\nbest silhouette among K = 3, 4, 5 is K =",
      int(table[table["K"].isin([3, 4, 5])].sort_values("silhouette").iloc[-1]["K"]))
