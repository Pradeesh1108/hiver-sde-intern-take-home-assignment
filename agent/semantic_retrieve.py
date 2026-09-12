# ============================================================
# agent/retriever_semantic.py — Semantic Retrieval (RAG)
# ============================================================
# Drop-in replacement for retrieve_by_keywords() in retriever.py
#
# How it works:
#   1. Embed the new customer message → query vector
#   2. Load stored embeddings for the classified intent
#   3. Cosine similarity → find top 3 most similar past threads
#   4. Return those threads' brand_replies
#
# Why only the classified intent's embeddings?
#   Intent filter first, then similarity search.
#   We never compare a battery complaint against autocorrect threads.
#   This reduces the search space from 80k to ~4k-14k per query.
#
# Files needed in datasets/semantic_index/:
#   <intent>.npy        — embeddings matrix (N × 768)
#   <intent>_meta.json  — thread_id + brand_replies
#
# Build the index first: python3 EDA/06_build_semantic_index.py
# ============================================================

import json
import os
import numpy as np
from functools import lru_cache

SEMANTIC_INDEX_DIR = "datasets/semantic_index"

# ─────────────────────────────────────────────
# LAZY LOADING — load per intent, cache in memory
# ─────────────────────────────────────────────
# We load one intent's data at a time and cache it.
# This avoids loading all 225MB at startup.
# The LRU cache holds the last 10 intents (one per intent).

_model = None

def _get_model():
    """Load and cache the embedding model. Loads once on first call."""
    global _model
    if _model is None:
        try:
            import os
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Explicitly detect GPU for faster inference
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
            _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers\n"
                "Or use retrieve_by_keywords() instead."
            )
    return _model


@lru_cache(maxsize=11)  # cache all 10 intents + 1 buffer
def _load_intent_index(intent: str):
    """
    Load embeddings and metadata for one intent.
    Cached — only reads from disk once per intent per session.

    Returns:
        (embeddings_matrix, metadata_list) or (None, None) if not found
    """
    npy_path  = os.path.join(SEMANTIC_INDEX_DIR, f"{intent}.npy")
    meta_path = os.path.join(SEMANTIC_INDEX_DIR, f"{intent}_meta.json")

    if not os.path.exists(npy_path) or not os.path.exists(meta_path):
        return None, None

    embeddings = np.load(npy_path)    # shape: (N, 768)

    with open(meta_path) as f:
        metadata = json.load(f)       # list of {thread_id, brand_replies}

    return embeddings, metadata


def retrieve_semantic(message: str, intent: str, n: int = 3) -> list:
    """
    Find the n most semantically similar past threads for this intent.

    Process:
      1. Embed the customer message
      2. Load stored embeddings for the classified intent
      3. Cosine similarity (dot product on unit vectors)
      4. Return top n threads with their brand_replies

    Args:
        message: cleaned customer tweet text
        intent:  classified intent string
        n:       number of threads to return (default 3)

    Returns:
        List of dicts with keys: thread_id, customer_msg, brand_replies
        Falls back to keyword retrieval if semantic index not available.

    Example:
        >>> results = retrieve_semantic(
        ...     "my iPhone keeps changing the letter i to a question mark",
        ...     "autocorrect_i_bug",
        ...     n=3
        ... )
        >>> results[0]["brand_replies"][0]
        "We'd like to look into this with you. Which model do you have..."
    """
    # Load the semantic index for this intent
    embeddings, metadata = _load_intent_index(intent)

    if embeddings is None:
        # Semantic index not built yet — fall back to keyword retrieval
        print(f"  [retriever] Semantic index not found for {intent}, "
              f"falling back to keyword retrieval")
        # Import here to avoid circular import
        from agent.retriever import retrieve_by_keywords
        return retrieve_by_keywords(message, intent, n)

    # Get embedding model
    try:
        model = _get_model()
    except ImportError as e:
        print(f"  [retriever] {e}")
        from agent.retriever import retrieve_by_keywords
        return retrieve_by_keywords(message, intent, n)

    # Embed the new customer message
    # normalize_embeddings=True → unit vector → cosine = dot product
    query_vector = model.encode(
        [message],
        normalize_embeddings = True,
        convert_to_numpy     = True,
        show_progress_bar    = False,
    )[0]  # shape: (768,)

    # Cosine similarity against all stored embeddings for this intent
    # Since both are unit normalised: similarity = dot product
    # embeddings shape: (N, 768), query shape: (768,)
    # Result shape: (N,) — one score per stored thread
    similarities = embeddings @ query_vector

    # Get top n indices by similarity score
    n_available = min(n, len(similarities))
    top_indices = similarities.argsort()[-n_available:][::-1]

    # Build result list
    results = []
    for idx in top_indices:
        entry = metadata[idx]
        results.append({
            "thread_id":     entry["thread_id"],
            "customer_msg":  entry["customer_msg"],
            "brand_replies": entry["brand_replies"],
            "similarity":    float(similarities[idx]),  # for debugging
        })

    return results


# ─────────────────────────────────────────────
# QUICK TEST
# python3 -m agent.retriever_semantic
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("my iPhone keeps changing the letter i to a weird box", "autocorrect_i_bug"),
        ("battery drops 30 percent in ten minutes its ridiculous", "battery_drain"),
        ("wifi keeps turning itself back on after I disable it",  "wifi_bluetooth"),
        ("iMessage not delivering to one specific person only",   "app_issue"),
    ]

    print("Testing semantic retrieval...\n")

    for message, intent in test_cases:
        print(f"  Intent: {intent}")
        print(f"  Query:  \"{message}\"")

        results = retrieve_semantic(message, intent, n=3)

        if not results:
            print("  No results returned")
        else:
            for i, r in enumerate(results, 1):
                sim = r.get("similarity", 0)
                print(f"  [{i}] similarity={sim:.3f}")
                print(f"       customer: \"{r['customer_msg'][:70]}\"")
                print(f"       reply:    \"{r['brand_replies'][0][:70]}\"")
        print()