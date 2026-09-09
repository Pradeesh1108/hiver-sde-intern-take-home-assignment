# ============================================================
# agent/retriever.py — Historical Thread Retrieval
# ============================================================
# One public function: retrieve(intent, n) → list of threads
#
# How it works:
#   1. Load the retrieval index (80k threads with keyword_hint)
#   2. Filter to threads matching the given intent
#   3. Return n random threads from that pool
#
# Why random and not similarity-based?
#   Embedding similarity would be more accurate — a message about
#   "battery dying at 80%" would find threads about "battery
#   dropping suddenly" rather than just any battery thread.
#   But it requires running all 80k messages through an embedding
#   model and storing vectors. For this project, keyword-filtered
#   random sampling gives good enough grounding with zero extra
#   infrastructure.
#   This is documented as a known limitation in the decision log
#   and report (D12). "One more week" would add embeddings.
# ============================================================

import json
import random
from functools import lru_cache


# ─────────────────────────────────────────────
# LOAD RETRIEVAL INDEX
# ─────────────────────────────────────────────
# @lru_cache means this function's result is cached after the
# first call. The 80k-entry JSON is only read from disk once,
# not on every retrieve() call.

@lru_cache(maxsize=1)
def _load_index() -> list:
    """Load and cache the retrieval index from disk."""
    with open("datasets/retrieval_index.json", "r") as f:
        return json.load(f)


def retrieve(intent: str, n: int = 3, seed: int = None) -> list:
    """
    Find historical Apple Support threads matching the given intent.

    Args:
        intent: classified intent string (e.g. "battery_drain")
        n:      number of threads to return (default 3)
        seed:   random seed for reproducibility (useful in eval)

    Returns:
        List of up to n thread dicts, each with:
            thread_id    — unique id
            customer_msg — opening customer message
            brand_replies — list of Apple's replies in this thread
            num_turns    — how long the conversation was

        Returns empty list if intent has no matching threads.

    Example:
        >>> threads = retrieve("battery_drain", n=3)
        >>> threads[0]["brand_replies"][0]
        "We're sorry to hear about the battery issue. What iOS version..."
    """
    index = _load_index()

    # Filter to threads whose keyword_hint matches the intent
    # keyword_hint was assigned by the same keyword logic used
    # during sampling — so it's consistent with the intent names
    matching = [
        entry for entry in index
        if entry.get("keyword_hint") == intent
    ]

    if not matching:
        # No threads found for this intent — return empty list
        # reply_drafter.py handles the empty case gracefully
        print(f"  [retriever] No threads found for intent: {intent}")
        return []

    # Sample n threads randomly from the matching pool
    # If fewer than n are available, return all of them
    if seed is not None:
        random.seed(seed)

    sample_size = min(n, len(matching))
    sampled = random.sample(matching, sample_size)

    return sampled


def retrieve_by_keywords(message: str, intent: str, n: int = 3) -> list:
    """
    Retrieve threads that match BOTH the intent AND contain
    at least one word from the customer message.

    This gives slightly better grounding than pure random sampling
    within an intent bucket — threads that share vocabulary with
    the customer message are more likely to be relevant.

    Args:
        message: the customer's cleaned message text
        intent:  classified intent string
        n:       number of threads to return

    Returns:
        List of up to n matching thread dicts.
        Falls back to random retrieve() if no keyword overlap found.
    """
    index = _load_index()

    # Filter by intent first
    intent_matches = [
        entry for entry in index
        if entry.get("keyword_hint") == intent
    ]

    if not intent_matches:
        return []

    # Extract meaningful words from the customer message
    # Filter out very short words (stop words like "my", "is", "the")
    message_words = set(
        word.lower()
        for word in message.split()
        if len(word) > 3
    )

    if not message_words:
        # Message too short to extract keywords — fall back to random
        return retrieve(intent, n)

    # Score each thread by how many message words appear in it
    scored = []
    for entry in intent_matches:
        entry_text = entry.get("customer_msg", "").lower()
        overlap = sum(1 for word in message_words if word in entry_text)
        if overlap > 0:
            scored.append((overlap, entry))

    if scored:
        # Sort by overlap score descending, take top n
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:n]]
    else:
        # No keyword overlap found — fall back to random sampling
        return retrieve(intent, n)


# ─────────────────────────────────────────────
# QUICK TEST
# python3 -m agent.retriever
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing retriever...\n")

    for intent in ["battery_drain", "ios_update_issue", "account_access"]:
        threads = retrieve(intent, n=2)
        print(f"  [{intent}] — {len(threads)} threads retrieved")
        for t in threads:
            print(f"    Customer: \"{t['customer_msg'][:70]}\"")
            if t["brand_replies"]:
                print(f"    Apple:    \"{t['brand_replies'][0][:70]}\"")
        print()

    print("Testing keyword retrieval...")
    threads = retrieve_by_keywords(
        "my battery is draining really fast since I updated",
        "ios_update_issue",
        n=3
    )
    print(f"  Retrieved {len(threads)} threads with keyword overlap")
    for t in threads:
        print(f"    \"{t['customer_msg'][:70]}\"")