# ============================================================
# EDA/01_extract_messages.py
# ============================================================
# What this does:
#   Reads apple_threads.json (80k threads)
#   Extracts the first customer message from each thread
#   Cleans the text
#   Saves to EDA/data/messages.json
#
# Input:  datasets/apple_threads.json
# Output: EDA/data/messages.json
#
# Run: python3 EDA/01_extract_messages.py
# ============================================================

import json
import re
import os
import random

INPUT_PATH  = "datasets/apple_threads.json"
OUTPUT_DIR  = "EDA/data"
OUTPUT_PATH = "EDA/data/messages.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: TEXT CLEANING
# ─────────────────────────────────────────────
# Light cleaning only — remove noise, keep semantic content.
#
# REMOVE: @mentions, URLs, HTML entities, extra whitespace
# KEEP:   "battery", "iOS", "Apple ID", "password", emojis, numbers
#         These are the signals the embedding model needs.
#         Do NOT remove them as stopwords.

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\S+', '', text)          # remove @mentions
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = text.replace('&amp;', '&')
    text = text.replace('&gt;',  '>')
    text = text.replace('&lt;',  '<')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;',  "'")
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    return text.strip()


# ─────────────────────────────────────────────
# STEP 2: EXTRACT FIRST CUSTOMER MESSAGE
# ─────────────────────────────────────────────
# Why first message only?
#   At runtime the agent classifies the opening message.
#   Clustering should match the same conditions.
#   Later turns contain Apple's questions ("Which iOS version?")
#   which add noise — Apple's language, not the customer's problem.

def get_first_customer_message(thread: dict):
    for turn in thread.get("turns", []):
        if turn.get("role") == "customer":
            text = clean_text(turn.get("text", ""))
            if text:
                return text
    return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1: LOADING THREADS")
print("=" * 55)

with open(INPUT_PATH) as f:
    threads = json.load(f)

print(f"Loaded {len(threads)} threads")


print("\n" + "=" * 55)
print("STEP 2: EXTRACTING FIRST CUSTOMER MESSAGES")
print("=" * 55)

messages = []
skipped  = 0

for thread in threads:
    msg = get_first_customer_message(thread)

    if msg is None:
        skipped += 1
        continue

    # Skip messages too short to embed meaningfully
    # "help" or "fix this" has no semantic signal
    if len(msg.split()) < 3:
        skipped += 1
        continue

    messages.append({
        "thread_id": thread["thread_id"],
        "message":   msg,
        "num_turns": thread["num_turns"],
    })

print(f"Extracted:  {len(messages)} messages")
print(f"Skipped:    {skipped} (no customer turn or < 3 words)")
print(f"Keep rate:  {len(messages)/len(threads)*100:.1f}%")


print("\n" + "=" * 55)
print("STEP 3: MESSAGE LENGTH STATS")
print("=" * 55)

lengths = sorted([len(m["message"].split()) for m in messages])
n       = len(lengths)

print(f"Min words:    {lengths[0]}")
print(f"Median words: {lengths[n//2]}")
print(f"Max words:    {lengths[-1]}")
print(f"Avg words:    {sum(lengths)/n:.1f}")

buckets = [
    ("3-10 words",  3,  10),
    ("11-20 words", 11, 20),
    ("21-30 words", 21, 30),
    ("31+ words",   31, 9999),
]
for label, lo, hi in buckets:
    count = sum(1 for l in lengths if lo <= l <= hi)
    pct   = count / n * 100
    bar   = "█" * int(pct / 2)
    print(f"  {label:<15} {count:>6} ({pct:4.1f}%)  {bar}")


print("\n" + "=" * 55)
print("STEP 4: SAMPLE MESSAGES (sanity check)")
print("=" * 55)

random.seed(42)
for i, m in enumerate(random.sample(messages, 8), 1):
    print(f"  [{i}] thread={m['thread_id']}  turns={m['num_turns']}")
    print(f"      \"{m['message'][:85]}\"")
    print()


print("=" * 55)
print("STEP 5: SAVING")
print("=" * 55)

with open(OUTPUT_PATH, "w") as f:
    json.dump(messages, f, indent=2)

print(f"Saved {len(messages)} messages → {OUTPUT_PATH}")
print()
print("Next: python3 EDA/02_embed.py")