# ============================================================
# EDA/02_embed.py
# ============================================================
# What this does:
#   Loads EDA/data/messages.json (79,937 messages)
#   Filters out non-English messages
#   Converts each message to a 768-dim semantic vector
#   Using sentence-transformers/all-mpnet-base-v2
#   Saves embeddings as EDA/data/embeddings.npy
#   Saves filtered messages as EDA/data/messages_filtered.json
#
# Input:  EDA/data/messages.json
# Output: EDA/data/embeddings.npy        (N x 768 float32 matrix)
#         EDA/data/messages_filtered.json (filtered message list)
#
# Install:
#   pip install sentence-transformers langdetect numpy
#
# Runtime: ~45-90 minutes on CPU (MacBook Air)
#          ~8-12 minutes on GPU
#          Progress bar shows estimated time remaining.
#
# Run: python3 EDA/02_embed.py
# ============================================================

import json
import os
import numpy as np

INPUT_MESSAGES  = "EDA/data/messages.json"
OUTPUT_EMBED    = "EDA/data/embeddings.npy"
OUTPUT_FILTERED = "EDA/data/messages_filtered.json"


# ─────────────────────────────────────────────
# STEP 1: LOAD MESSAGES
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1: LOADING MESSAGES")
print("=" * 55)

with open(INPUT_MESSAGES) as f:
    messages = json.load(f)

print(f"Loaded {len(messages)} messages")


# ─────────────────────────────────────────────
# STEP 2: FILTER NON-ENGLISH
# ─────────────────────────────────────────────
# It is not perfect (short messages are hard to detect)
# so we keep messages where detection is uncertain.
#
# Strategy:
#   Detected English → keep
#   Detected non-English confidently → discard
#   Detection failed or uncertain → keep (safer to include)

print("\n" + "=" * 55)
print("STEP 2: FILTERING NON-ENGLISH MESSAGES")
print("=" * 55)

try:
    from langdetect import detect, LangDetectException
    langdetect_available = True
    print("langdetect available — filtering non-English messages")
except ImportError:
    langdetect_available = False
    print("langdetect not installed — skipping language filter")
    print("Install with: pip install langdetect")
    print("Continuing with all messages...")

filtered_messages = []
removed_count     = 0

if langdetect_available:
    for i, item in enumerate(messages):
        try:
            lang = detect(item["message"])
            if lang == "en":
                filtered_messages.append(item)
            else:
                removed_count += 1
        except LangDetectException:
            # Detection failed — keep the message
            # Short messages often fail detection
            filtered_messages.append(item)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(messages)}...")
else:
    filtered_messages = messages

print(f"\nKept:    {len(filtered_messages)} messages")
print(f"Removed: {removed_count} non-English messages")
print(f"({removed_count/len(messages)*100:.1f}% of total)")

# Save filtered messages
with open(OUTPUT_FILTERED, "w") as f:
    json.dump(filtered_messages, f, indent=2)
print(f"Saved filtered messages → {OUTPUT_FILTERED}")


# ─────────────────────────────────────────────
# STEP 3: LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────
# Model: all-mpnet-base-v2
#   - 768 dimensions (vs 384 for MiniLM — richer representation)
#   - Strong on short text (tweets are short)
#   - First run: downloads ~420MB model weights automatically
#   - Subsequent runs: loads from cache (instant)
#
# normalize_embeddings=True:
#   Normalises each vector to unit length.
#   This makes cosine similarity = dot product, which is
#   faster and required for UMAP with cosine metric.

print("\n" + "=" * 55)
print("STEP 3: LOADING EMBEDDING MODEL")
print("=" * 55)

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    raise

# Explicitly detect GPU for faster embeddings
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
else:
    device = "cpu"

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
print(f"Model loaded: all-mpnet-base-v2 (768 dimensions) on {device.upper()}")


# ─────────────────────────────────────────────
# STEP 4: GENERATE EMBEDDINGS
# ─────────────────────────────────────────────
# batch_size=64:
#   Process 64 messages at a time.
#   Higher = faster but uses more RAM.
#   64 is safe for 8GB RAM MacBook Air.
#   Reduce to 32 if you get memory errors.
#
# show_progress_bar=True:
#   Shows estimated time remaining.
#   On MacBook Air CPU: ~45-90 minutes for 75k messages.
#
# normalize_embeddings=True:
#   Unit-length vectors. Required for cosine UMAP later.

print("\n" + "=" * 55)
print("STEP 4: GENERATING EMBEDDINGS")
print("=" * 55)
print(f"Embedding {len(filtered_messages)} messages...")
if device == "cpu":
    print("This takes 45-90 minutes on CPU. Progress bar below:\n")
else:
    print(f"This should be fast using {device.upper()} acceleration. Progress bar below:\n")

texts = [item["message"] for item in filtered_messages]

embeddings = model.encode(
    texts,
    batch_size          = 64,
    show_progress_bar   = True,
    normalize_embeddings= True,
    convert_to_numpy    = True,
)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"  Rows:    {embeddings.shape[0]} (one per message)")
print(f"  Columns: {embeddings.shape[1]} (768 dimensions)")
print(f"  dtype:   {embeddings.dtype}")
print(f"  Size:    {embeddings.nbytes / 1024**2:.1f} MB in memory")


# ─────────────────────────────────────────────
# STEP 5: SANITY CHECK
# ─────────────────────────────────────────────
# Verify embeddings look reasonable before saving.
# Two similar messages should have high cosine similarity.
# Two different messages should have low cosine similarity.

print("\n" + "=" * 55)
print("STEP 5: SANITY CHECK")
print("=" * 55)

# Find two similar messages and compute their similarity
# Since embeddings are normalised, dot product = cosine similarity
similar_pairs = [
    (0, 1),   # whatever the first two messages happen to be
]

# Also pick some known-similar messages manually
sample_texts = [
    "my battery drains really fast",
    "battery dies in 2 hours",
    "can't log into my Apple ID",
]
sample_embeddings = model.encode(
    sample_texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

# Compute pairwise similarities
print("Similarity between similar messages (should be HIGH > 0.7):")
sim_battery = float(np.dot(sample_embeddings[0], sample_embeddings[1]))
print(f"  'battery drains fast' vs 'battery dies in 2 hours':  {sim_battery:.3f}")

print("\nSimilarity between different messages (should be LOW < 0.5):")
sim_diff = float(np.dot(sample_embeddings[0], sample_embeddings[2]))
print(f"  'battery drains fast' vs 'cant log into Apple ID':   {sim_diff:.3f}")

# Value ranges
print(f"\nEmbedding value ranges:")
print(f"  Min: {embeddings.min():.4f}")
print(f"  Max: {embeddings.max():.4f}")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  All norms ≈ 1.0: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")


# ─────────────────────────────────────────────
# STEP 6: SAVE
# ─────────────────────────────────────────────
# Save as .npy (numpy binary format) not JSON.
# Reason: 75k × 768 floats as JSON would be ~1.5GB.
#         As .npy it is ~220MB and loads in seconds.
# Loading later: embeddings = np.load("EDA/data/embeddings.npy")

print("\n" + "=" * 55)
print("STEP 6: SAVING")
print("=" * 55)

np.save(OUTPUT_EMBED, embeddings)

file_size = os.path.getsize(OUTPUT_EMBED) / 1024**2
print(f"Saved embeddings → {OUTPUT_EMBED}")
print(f"  Shape: {embeddings.shape}")
print(f"  File size: {file_size:.1f} MB")
print()
print("Next: python3 EDA/03_cluster.py")