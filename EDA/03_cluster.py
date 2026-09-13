# ============================================================
# EDA/03_cluster.py
# ============================================================
# What this does:
#   Loads embeddings.npy (76,656 x 768)
#   Reduces dimensions: 768 → 30 using UMAP
#   Clusters reduced vectors using HDBSCAN
#   Saves cluster assignments and UMAP embeddings
#
# Input:  EDA/data/embeddings.npy
#         EDA/data/messages_filtered.json
# Output: EDA/data/umap_embeddings.npy
#         EDA/data/cluster_assignments.csv
#
# Install:
#   pip install umap-learn hdbscan pandas
#
# Runtime:
#   UMAP:    20-35 minutes on CPU for 76k messages
#   HDBSCAN: 3-8 minutes
#
# NOTE: UMAP output is saved so if you want to re-run HDBSCAN
#       with different parameters, use --skip-umap flag:
#       python3 EDA/03_cluster.py --skip-umap
#
# Run: python3 EDA/03_cluster.py
# ============================================================

import json
import os
import sys
import random
import numpy as np
import pandas as pd
from collections import Counter

INPUT_EMBED     = "EDA/data/embeddings.npy"
INPUT_MESSAGES  = "EDA/data/messages_filtered.json"
OUTPUT_UMAP     = "EDA/data/umap_embeddings.npy"
OUTPUT_CLUSTERS = "EDA/data/cluster_assignments.csv"

# Check if --skip-umap flag is passed
# Use this if UMAP was already run and you only want to re-tune HDBSCAN
SKIP_UMAP = "--skip-umap" in sys.argv


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1: LOADING DATA")
print("=" * 55)

embeddings = np.load(INPUT_EMBED)
print(f"Embeddings: {embeddings.shape}  ({embeddings.nbytes/1024**2:.0f} MB)")

with open(INPUT_MESSAGES) as f:
    messages = json.load(f)
print(f"Messages:   {len(messages)}")

assert len(messages) == embeddings.shape[0], (
    f"Count mismatch: {len(messages)} messages but {embeddings.shape[0]} embeddings.\n"
    f"Re-run 02_embed.py to fix this."
)
print("Count check: OK")


# ─────────────────────────────────────────────
# STEP 2: UMAP
#   n_components=30  output dimensions
#                    NOT 2 (that's for plotting only)
#                    30D gives HDBSCAN much more to work with
#
#   metric="cosine"  right measure for normalised embeddings
#
#   min_dist=0.0     pack nearby points as tight as possible
#                    better for clustering (want dense groups)
#
#   random_state=42  makes results reproducible

print("\n" + "=" * 55)
if SKIP_UMAP:
    print("STEP 2: LOADING EXISTING UMAP EMBEDDINGS (--skip-umap)")
    print("=" * 55)
    umap_embeddings = np.load(OUTPUT_UMAP)
    print(f"Loaded: {umap_embeddings.shape}")
else:
    print("STEP 2: UMAP DIMENSIONALITY REDUCTION")
    print("=" * 55)
    print(f"768 → 30 dimensions  |  {len(messages)} messages")
    print("Takes 20-35 minutes on CPU...\n")

    try:
        import umap.umap_ as umap
    except ImportError:
        print("umap-learn not installed. Run: pip install umap-learn")
        raise

    reducer = umap.UMAP(
        n_neighbors  = 30,
        n_components = 30,
        metric       = "cosine",
        min_dist     = 0.0,
        random_state = 42,
        verbose      = True,
    )

    umap_embeddings = reducer.fit_transform(embeddings)
    print(f"\nOutput shape: {umap_embeddings.shape}")

    np.save(OUTPUT_UMAP, umap_embeddings)
    print(f"Saved → {OUTPUT_UMAP}")
    print("(Use --skip-umap on next run to reuse this)")


# ─────────────────────────────────────────────
# STEP 3: HDBSCAN CLUSTERING
# Parameters:
#   min_cluster_size=100
#     Minimum messages to form a cluster.
#     100 = ~0.13% of 76k messages.
#     Catches rare but real issue types.
#     Increase to 200 if you get too many clusters (30+).
#     Decrease to 50 if too few clusters (under 8).
#
#   min_samples=20
#     Controls conservativeness.
#     Higher = fewer clusters, more noise.
#     Lower  = more clusters, less noise.
#
#   metric="euclidean"
#     After UMAP we use euclidean (UMAP converts cosine → euclidean)
#
#   cluster_selection_method="eom"
#     Excess of Mass — finds compact, well-separated clusters.
#     Better than "leaf" for our use case.

print("\n" + "=" * 55)
print("STEP 3: HDBSCAN CLUSTERING")
print("=" * 55)

try:
    import hdbscan
except ImportError:
    print("hdbscan not installed. Run: pip install hdbscan")
    raise

MIN_CLUSTER_SIZE = 100
MIN_SAMPLES      = 20

print(f"min_cluster_size = {MIN_CLUSTER_SIZE}")
print(f"min_samples      = {MIN_SAMPLES}")
print("Running...\n")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size       = MIN_CLUSTER_SIZE,
    min_samples            = MIN_SAMPLES,
    metric                 = "euclidean",
    cluster_selection_method = "eom",
    prediction_data        = True,
)

labels = clusterer.fit_predict(umap_embeddings)


# ─────────────────────────────────────────────
# STEP 4: RESULTS
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 4: RESULTS")
print("=" * 55)

n_total    = len(labels)
n_noise    = int((labels == -1).sum())
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"Total messages:   {n_total:>7}")
print(f"Clusters found:   {n_clusters:>7}")
print(f"Noise points:     {n_noise:>7}  ({n_noise/n_total*100:.1f}%)")
print(f"Clustered points: {n_total-n_noise:>7}  ({(n_total-n_noise)/n_total*100:.1f}%)")

# Cluster sizes sorted by size descending
label_counts = Counter(labels)
sorted_clusters = sorted(
    [(l, c) for l, c in label_counts.items() if l != -1],
    key=lambda x: -x[1]
)

print(f"\n{'Cluster':<10} {'Size':>8} {'%':>7}  {'Bar'}")
print("─" * 50)
for label, count in sorted_clusters:
    pct = count / n_total * 100
    bar = "█" * int(pct * 1.5)
    print(f"Cluster {label:<4} {count:>8} ({pct:5.1f}%)  {bar}")

if -1 in label_counts:
    nc = label_counts[-1]
    print(f"{'Noise':<10} {nc:>8} ({nc/n_total*100:5.1f}%)")

sizes = [c for _, c in sorted_clusters]
print(f"\nLargest cluster:  {max(sizes)}")
print(f"Smallest cluster: {min(sizes)}")
print(f"Median size:      {sorted(sizes)[len(sizes)//2]}")


# ─────────────────────────────────────────────
# STEP 5: SAVE CLUSTER ASSIGNMENTS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 5: SAVING")
print("=" * 55)

rows = [{
    "thread_id":     item["thread_id"],
    "message":       item["message"],
    "cluster_label": int(labels[i]),
    "num_turns":     item.get("num_turns", 0),
} for i, item in enumerate(messages)]

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CLUSTERS, index=False)
print(f"Saved {len(df)} rows → {OUTPUT_CLUSTERS}")


# ─────────────────────────────────────────────
# STEP 6: QUICK PEEK — 3 samples per cluster
# ─────────────────────────────────────────────
# Just enough to see what each cluster is about.
# Full inspection happens in 04_inspect.py.

print("\n" + "=" * 55)
print("STEP 6: QUICK PEEK — 3 SAMPLES PER CLUSTER")
print("=" * 55)

random.seed(42)
msg_by_cluster = {}
for row in rows:
    l = row["cluster_label"]
    if l not in msg_by_cluster:
        msg_by_cluster[l] = []
    msg_by_cluster[l].append(row["message"])

for label, count in sorted_clusters:
    sample = random.sample(msg_by_cluster[label], min(3, count))
    print(f"\nCluster {label}  ({count} messages)")
    for msg in sample:
        print(f"  • {msg[:85]}")

noise_msgs = msg_by_cluster.get(-1, [])
if noise_msgs:
    print(f"\nNoise  ({len(noise_msgs)} messages — sample of 3)")
    sample = random.sample(noise_msgs, min(3, len(noise_msgs)))
    for msg in sample:
        print(f"  • {msg[:85]}")

print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"  {n_clusters} clusters discovered")
print(f"  {n_noise} noise points ({n_noise/n_total*100:.1f}%) → will become general_inquiry")
print()
print("If the cluster count looks wrong:")
print("  Too many clusters (>30): increase min_cluster_size to 200")
print("  Too few clusters (<8):   decrease min_cluster_size to 50")
print("  Then run: python3 EDA/03_cluster.py --skip-umap")
print()
print("Next: python3 EDA/04_inspect.py")