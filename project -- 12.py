"""
Unsupervised Clustering Pipeline (robust version)
====================================================
Reads a feature-engineered CSV, finds the optimal k, and compares
KMeans / Hierarchical / DBSCAN to pick the best clustering model.

Hierarchical clustering is wrapped in a try/except: if scipy's
compiled hierarchy module is blocked (e.g. by Windows Application
Control / Smart App Control / a corporate WDAC policy), this script
will skip it and continue with KMeans + DBSCAN instead of crashing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
DATA_PATH = "final_feature_engineered.csv"   # <-- change if needed
df = pd.read_csv(DATA_PATH)

LABEL_COL = "Category"
DROP_COLS = ["URL", "Title", "title_clean", LABEL_COL, "url_year", "url_month"]

feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].copy()

for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)

y_true = LabelEncoder().fit_transform(df[LABEL_COL])   # for evaluation only

# ---------------------------------------------------------------
# 2. Scale + reduce dimensionality
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced from {X.shape[1]} -> {X_pca.shape[1]} dimensions "
      f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance kept)")

# ---------------------------------------------------------------
# 3. KMeans — sweep k, collect metrics
# ---------------------------------------------------------------
k_range = range(20, 50)
kmeans_results = []
kmeans_models = {}

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)
    kmeans_models[k] = km
    kmeans_results.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X_pca, labels),
        "davies_bouldin": davies_bouldin_score(X_pca, labels),
        "calinski_harabasz": calinski_harabasz_score(X_pca, labels),
        "cluster_sizes": np.bincount(labels).tolist(),
    })

kmeans_df = pd.DataFrame(kmeans_results)
print("\n--- KMeans ---")
print(kmeans_df.drop(columns="cluster_sizes").round(4))

# ---------------------------------------------------------------
# 4. Hierarchical (Ward) — wrapped so a blocked DLL can't crash the run
# ---------------------------------------------------------------
hier_df = None
try:
    from sklearn.cluster import AgglomerativeClustering

    hier_results = []
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X_pca)
        hier_results.append({
            "k": k,
            "silhouette": silhouette_score(X_pca, labels),
            "davies_bouldin": davies_bouldin_score(X_pca, labels),
            "calinski_harabasz": calinski_harabasz_score(X_pca, labels),
            "cluster_sizes": np.bincount(labels).tolist(),
        })
    hier_df = pd.DataFrame(hier_results)
    print("\n--- Hierarchical (Ward) ---")
    print(hier_df.drop(columns="cluster_sizes").round(4))
    print("Cluster sizes per k:", dict(zip(hier_df["k"], hier_df["cluster_sizes"])))

except Exception as e:
    print("\n--- Hierarchical (Ward): SKIPPED ---")
    print(f"Reason: {type(e).__name__}: {e}")
    print("This is almost always a Windows Application Control / antivirus block "
          "on scipy's compiled hierarchy module, not a bug in this script. "
          "Continuing with KMeans + DBSCAN only.")

# ---------------------------------------------------------------
# 5. DBSCAN — pick eps via k-distance graph, then sweep
# ---------------------------------------------------------------
min_samples = 10
nn = NearestNeighbors(n_neighbors=min_samples).fit(X_pca)
distances, _ = nn.kneighbors(X_pca)
k_dist_sorted = np.sort(distances[:, -1])
# Plot k_dist_sorted yourself (plt.plot(k_dist_sorted)) to eyeball the "knee".

dbscan_results = []
for eps in [4, 5, 6, 7, 8, 9, 10, 12]:
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_pca)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    if n_clusters >= 2:
        mask = labels != -1
        sil = silhouette_score(X_pca[mask], labels[mask]) if mask.sum() > 1 else np.nan
    else:
        sil = np.nan
    dbscan_results.append({"eps": eps, "n_clusters": n_clusters,
                            "noise_pct": round(noise_pct, 1), "silhouette": sil})

dbscan_df = pd.DataFrame(dbscan_results)
print("\n--- DBSCAN ---")
print(dbscan_df.round(4))

# ---------------------------------------------------------------
# 6. Select the best model
# ---------------------------------------------------------------
# Don't just chase the highest silhouette score — check cluster_sizes too.
# A high score with one giant cluster + tiny 1-2% pockets means the
# algorithm found outliers, not real structure.
BEST_K = 3
final_model = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = final_model.fit_predict(X_pca)

print("\nFinal model: KMeans, k =", BEST_K)
print("Final cluster sizes:", np.bincount(df["cluster"]))
print("Silhouette (final):", silhouette_score(X_pca, df["cluster"]))

# ---------------------------------------------------------------
# 7. Evaluate against known Category labels (sanity check only)
# ---------------------------------------------------------------
ari = adjusted_rand_score(y_true, df["cluster"])
nmi = normalized_mutual_info_score(y_true, df["cluster"])
print(f"\nARI vs true Category: {ari:.4f}")
print(f"NMI vs true Category: {nmi:.4f}")
print("\nCrosstab (cluster vs actual Category):")
print(pd.crosstab(df["cluster"], df[LABEL_COL]))

# ---------------------------------------------------------------
# 8. Profile the clusters
# ---------------------------------------------------------------
profile_cols = ["domain_freq_encoded", "title_char_len", "title_word_count",
                 "kw_markets", "kw_business", "kw_technology", "kw_politics",
                 "kw_health", "kw_energy"]
print("\nCluster profiles (feature means):")
print(df.groupby("cluster")[profile_cols].mean().round(3))




# Visualization of Result

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Clustering Analysis — Optimal k & Model Comparison", fontsize=15, fontweight="bold")

# --- Plot 1: Elbow curve (KMeans inertia) ---
ax = axes[0, 0]
ax.plot(kmeans_df["k"], kmeans_df["inertia"], marker="o", color="#2563eb")
ax.set_title("Elbow Method (KMeans)")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia (WCSS)")
ax.axvline(BEST_K, color="red", linestyle="--", alpha=0.6, label=f"chosen k={BEST_K}")
ax.legend()

# --- Plot 2: Silhouette score comparison ---
ax = axes[0, 1]
ax.plot(kmeans_df["k"], kmeans_df["silhouette"], marker="o", label="KMeans", color="#2563eb")
if hier_df is not None:
    ax.plot(hier_df["k"], hier_df["silhouette"], marker="s", label="Hierarchical (Ward)", color="#16a34a")
ax.set_title("Silhouette Score vs k")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.axvline(BEST_K, color="red", linestyle="--", alpha=0.6)
ax.legend()

# --- Plot 3: Davies-Bouldin comparison (lower = better) ---
ax = axes[0, 2]
ax.plot(kmeans_df["k"], kmeans_df["davies_bouldin"], marker="o", label="KMeans", color="#2563eb")
if hier_df is not None:
    ax.plot(hier_df["k"], hier_df["davies_bouldin"], marker="s", label="Hierarchical (Ward)", color="#16a34a")
ax.set_title("Davies-Bouldin Index vs k (lower is better)")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Davies-Bouldin Index")
ax.legend()

# --- Plot 4: DBSCAN k-distance graph (for eps selection) ---
ax = axes[1, 0]
ax.plot(k_dist_sorted, color="#9333ea")
ax.set_title(f"k-Distance Graph (min_samples={min_samples})")
ax.set_xlabel("Points sorted by distance")
ax.set_ylabel(f"Distance to {min_samples}th nearest neighbor")

# --- Plot 5: Final cluster sizes (bar chart) ---
ax = axes[1, 1]
sizes = np.bincount(df["cluster"])
colors = ["#2563eb", "#16a34a", "#f97316", "#dc2626", "#9333ea"]
bars = ax.bar(range(len(sizes)), sizes, color=colors[:len(sizes)])
ax.set_title(f"Final Cluster Sizes (KMeans, k={BEST_K})")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of articles")
ax.set_xticks(range(len(sizes)))
for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, size + 20, str(size), ha="center", fontsize=9)

# --- Plot 6: 2D PCA scatter colored by final cluster ---
ax = axes[1, 2]
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca_2d.fit_transform(X_scaled)
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="tab10", s=8, alpha=0.6)
ax.set_title("Clusters in 2D (PCA projection)")
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
ax.add_artist(legend1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("clustering_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to clustering_analysis.png")
plt.show()