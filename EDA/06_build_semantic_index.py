# ============================================================
# EDA/06_build_semantic_index.py — Build Semantic Index
# ============================================================
# What this does:
#   Reads retrieval_index.json (80,483 threads)
#   Embeds every customer_msg using all-mpnet-base-v2
#   Groups embeddings by intent
#   Saves one .npy file per intent (the search keys)
#   Saves one _meta.json per intent (thread_id + brand_replies)
#
# At retrieval time:
#   1. Embed new customer message → query vector
#   2. Load embeddings for the classified intent
#   3. Cosine similarity → find top 3 most similar
#   4. Return their brand_replies to the reply drafter
#
# Input:  EDA/data/retrieval_index.json
# Output: datasets/semantic_index/<intent>.npy
#         datasets/semantic_index/<intent>_meta.json
#
# Install: pip install sentence-transformers numpy
# Runtime: ~15-20 minutes on CPU for 80k messages
#
# Run: python3 EDA/06_build_semantic_index.py
# ============================================================

import json
import os
import numpy as np
from collections import defaultdict

INPUT_INDEX  = "EDA/data/retrieval_index.json"
OUTPUT_DIR   = "datasets/semantic_index"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: LOAD RETRIEVAL INDEX
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1: LOADING RETRIEVAL INDEX")
print("=" * 55)

with open(INPUT_INDEX) as f:
    index = json.load(f)

print(f"Total threads: {len(index)}")

# Show distribution by intent
from collections import Counter
dist = Counter(entry["intent"] for entry in index)
print("\nIntent distribution:")
for intent, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {intent:<25} {count:>6}")


# ─────────────────────────────────────────────
# STEP 2: GROUP BY INTENT
# ─────────────────────────────────────────────
# We process one intent at a time.
# This way we can embed ~4,000-14,000 messages per batch
# and save incrementally — if the script is interrupted,
# already-saved intents do not need to be re-embedded.

print("\n" + "=" * 55)
print("STEP 2: GROUPING BY INTENT")
print("=" * 55)

by_intent = defaultdict(list)
for entry in index:
    by_intent[entry["intent"]].append({
        "thread_id":    entry["thread_id"],
        "customer_msg": entry["customer_msg"],
        "brand_replies": entry["brand_replies"],
    })

intent_names = sorted(by_intent.keys())
print(f"Intents: {intent_names}")


# ─────────────────────────────────────────────
# STEP 3: LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────
# Same model used in EDA/02_embed.py for consistency.
# normalize_embeddings=True means cosine similarity = dot product.
# This makes similarity computation faster at query time.

print("\n" + "=" * 55)
print("STEP 3: LOADING EMBEDDING MODEL")
print("=" * 55)

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    raise

# Explicitly detect GPU for faster index building
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
print(f"Model loaded: all-mpnet-base-v2 (768 dimensions) on {device.upper()}")



# ─────────────────────────────────────────────
# STEP 4: EMBED AND SAVE PER INTENT
# ─────────────────────────────────────────────
# For each intent:
#   a) Collect all customer messages
#   b) Embed them in batches
#   c) Save embeddings as .npy  (the search key)
#   d) Save metadata as .json   (thread_id + brand_replies)
#
# Why split by intent?
#   At query time we only load one intent's embeddings.
#   Loading 4,077 battery_drain embeddings (~12MB) is much
#   faster than loading all 80k embeddings (~225MB).
#   Intent filtering happens BEFORE similarity search —
#   we never compare a battery complaint against autocorrect threads.

print("\n" + "=" * 55)
print("STEP 4: EMBEDDING BY INTENT")
print("=" * 55)

for intent in intent_names:
    entries   = by_intent[intent]
    npy_path  = os.path.join(OUTPUT_DIR, f"{intent}.npy")
    meta_path = os.path.join(OUTPUT_DIR, f"{intent}_meta.json")

    # Skip if already done (allows resuming interrupted runs)
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        print(f"  [{intent}] already done — skipping")
        continue

    print(f"\n  [{intent}]  {len(entries)} messages...")

    # Extract customer messages for embedding
    messages = [e["customer_msg"] for e in entries]

    # Embed all messages for this intent
    # batch_size=64 is safe for 8GB RAM
    embeddings = model.encode(
        messages,
        batch_size           = 64,
        show_progress_bar    = True,
        normalize_embeddings = True,  # unit vectors → cosine = dot product
        convert_to_numpy     = True,
    )

    print(f"  Embedding shape: {embeddings.shape}")

    # Save embeddings — the search key
    np.save(npy_path, embeddings)

    # Save metadata — thread_id + brand_replies (the payload)
    # We do NOT store customer_msg here — it is only needed for embedding
    # which is already done. We store brand_replies because that is what
    # the reply drafter needs.
    meta = [
        {
            "thread_id":    e["thread_id"],
            "customer_msg": e["customer_msg"],  # keep for debugging
            "brand_replies": e["brand_replies"],
        }
        for e in entries
    ]

    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    file_size = os.path.getsize(npy_path) / 1024**2
    print(f"  Saved: {npy_path}  ({file_size:.1f} MB)")
    print(f"  Saved: {meta_path}")


# ─────────────────────────────────────────────
# STEP 5: VERIFY
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 5: VERIFICATION")
print("=" * 55)

total_size = 0
for intent in intent_names:
    npy_path  = os.path.join(OUTPUT_DIR, f"{intent}.npy")
    meta_path = os.path.join(OUTPUT_DIR, f"{intent}_meta.json")

    emb  = np.load(npy_path)
    with open(meta_path) as f:
        meta = json.load(f)

    assert len(meta) == emb.shape[0], f"Count mismatch for {intent}"
    assert emb.shape[1] == 768, f"Wrong embedding dim for {intent}"

    file_size   = os.path.getsize(npy_path) / 1024**2
    total_size += file_size
    print(f"  [{intent}]  {emb.shape[0]} vectors  {file_size:.1f} MB  OK")

print(f"\n  Total size: {total_size:.1f} MB")
print(f"  (vs 225MB for full matrix — same total but loaded per-intent at runtime)")

print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"""
Semantic index saved to: datasets/semantic_index/

Files created per intent:
  <intent>.npy       — embeddings matrix (N × 768) — the search key
  <intent>_meta.json — thread_id + brand_replies    — the payload

Next: update agent/retriever.py to use retrieve_semantic()
      (see the updated retriever.py)
""")