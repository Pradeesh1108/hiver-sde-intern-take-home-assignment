# ============================================================
# EDA2/04_inspect.py — Cluster Inspection Report
# ============================================================
# What this does:
#   Reads cluster_assignments.csv
#   For each proposed merge group, shows:
#     - representative messages (closest to centroid)
#     - most common words
#     - sample Apple replies from those threads
#     - size and % of total
#   Saves a full report to EDA2/data/cluster_report.json
#   Also saves a readable cluster_report.txt
#
# Input:  EDA2/data/cluster_assignments.csv
#         EDA2/data/umap_embeddings.npy
#         EDA2/data/messages_filtered.json
#         datasets/apple_threads.json
#
# Output: EDA2/data/cluster_report.json
#         EDA2/data/cluster_report.txt
#
# Run: python3 EDA2/04_inspect.py
# ============================================================

import json
import os
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

INPUT_CLUSTERS  = "EDA2/data/cluster_assignments.csv"
INPUT_UMAP      = "EDA2/data/umap_embeddings.npy"
INPUT_MESSAGES  = "EDA2/data/messages_filtered.json"
INPUT_THREADS   = "datasets/apple_threads.json"
OUTPUT_JSON     = "EDA2/data/cluster_report.json"
OUTPUT_TXT      = "EDA2/data/cluster_report.txt"


# ─────────────────────────────────────────────
# MERGE MAP
# ─────────────────────────────────────────────
# Based on visual inspection of 03_cluster.py output.
# Each key is a proposed intent name.
# Value is the list of HDBSCAN cluster IDs to merge into it.
# Noise (-1) → general_inquiry automatically.

MERGE_MAP = {
    "autocorrect_i_bug": {
        "clusters": [38, 61, 29, 19, 10, 59, 53, 39, 62, 48, 47, 28, 67, 58, 63, 64],
        "description": (
            "Customer is experiencing the iOS 11 autocorrect bug where the letter "
            "I is replaced by A? or a question mark in a box. Also includes "
            "the related it to I.T and is to I.S autocorrect glitches. "
            "This is a specific known iOS 11 bug with a targeted fix."
        ),
    },
    "ios_update_general": {
        "clusters": [86, 104, 91, 35, 100, 81, 33, 105, 103, 101, 94,
                     77, 76, 2, 75, 66, 72, 45, 49],
        "description": (
            "Customer is experiencing general problems after installing an iOS or "
            "macOS update — slowness, crashes, features broken, lock screen issues, "
            "screen rotation problems, or overall dissatisfaction with a new OS version. "
            "Also includes macOS High Sierra and iPad Pro update issues. "
            "IMPORTANT: If the problem is specifically the I-bug use autocorrect_i_bug. "
            "If the problem is specifically battery drain use battery_drain. "
            "If the problem is specifically WiFi or Bluetooth use wifi_bluetooth."
        ),
    },
    "battery_drain": {
        "clusters": [68, 82, 78, 90, 89, 87, 85, 6],
        "description": (
            "Customer reports battery draining unusually fast, phone dying at high "
            "percentages, not charging properly, or battery life much shorter than "
            "expected. Includes both update-related battery drain and general battery "
            "issues. IMPORTANT: If battery drain started after an iOS update and "
            "the update is the main complaint, still use battery_drain if battery "
            "is clearly the primary symptom."
        ),
    },
    "wifi_bluetooth": {
        "clusters": [12, 14, 16, 36, 15],
        "description": (
            "Customer has connectivity problems — WiFi dropping or not connecting, "
            "Bluetooth turning on by itself or not pairing, CarPlay disconnecting. "
            "Usually caused by iOS 11 update changing connectivity behaviour. "
            "IMPORTANT: If the update itself is the main complaint and not "
            "connectivity specifically, use ios_update_general."
        ),
    },
    "phone_freezing": {
        "clusters": [92, 98, 94, 102, 99, 93, 97, 80],
        "description": (
            "Customer phone is freezing, showing a black screen, restarting "
            "randomly, or completely unresponsive and will not turn on. Device is "
            "non-functional or intermittently functional. IMPORTANT: If freezing "
            "started after a specific update and the update is the main complaint, "
            "use ios_update_general."
        ),
    },
    "account_access": {
        "clusters": [24, 25, 73, 26, 30, 31],
        "description": (
            "Customer cannot access their Apple ID, iCloud account, or App Store "
            "due to forgotten password, locked account, missing verification code, "
            "or two-factor authentication issues. Also includes phishing and scam "
            "emails pretending to be from Apple, iCloud storage questions, and "
            "iCloud backup problems. IMPORTANT: Do not use if the customer is "
            "having trouble with a specific third-party app login — use app_issue."
        ),
    },
    "device_hardware": {
        "clusters": [69, 70, 4, 79, 88, 32, 51, 9, 8, 80, 43, 96, 1],
        "description": (
            "Customer has a physical hardware problem — cracked or flickering screen, "
            "keyboard not working, broken or burnt charger, overheating, Touch ID or "
            "Face ID not recognising, camera broken, contacts deleted from device, "
            "Apple Watch hardware issues or WatchOS problems. "
            "IMPORTANT: If the hardware problem started after an update but the "
            "physical component is clearly broken, still use device_hardware."
        ),
    },
    "order_purchase": {
        "clusters": [74, 65],
        "description": (
            "Customer has a question or problem related to purchasing an Apple product, "
            "iPhone X reservation or delivery, billing charge, activation of a new "
            "device, or refund request. "
            "IMPORTANT: If the device arrived and has a hardware defect, use "
            "device_hardware instead."
        ),
    },
    "app_issue": {
        "clusters": [57, 52, 71, 41, 46, 50, 42, 22, 37, 17, 55, 23, 18, 11,
                     40, 60, 44, 56, 27, 3, 34, 21, 0, 20],
        "description": (
            "Customer is having a problem with a specific app — iMessage not "
            "delivering, App Store not loading, FaceTime dropping, Mail not sending, "
            "Siri broken, AirPods not working, speaker phone failing, Calendar bugs, "
            "Notes deleting, Podcast broken, Maps or GPS not working, screenshots "
            "failing, AirPlay or HomeKit not working, alarm silent, notifications "
            "not arriving. IMPORTANT: Do not use if the customer cannot log in "
            "due to Apple ID issues — use account_access. Do not use if the problem "
            "is clearly physical hardware — use device_hardware."
        ),
    },
    "general_inquiry": {
        "clusters": [-1, 83, 84, 7, 5, 54, 95],
        "description": (
            "Customer is making a vague complaint with no specific problem, asking "
            "a general customer service question, requesting a Genius Bar appointment, "
            "or the message is too short or unclear to classify into any specific "
            "intent. Use as last resort only when none of the above intents fit."
        ),
    },
}
INTENT_NAMES = list(MERGE_MAP.keys())


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

STOPWORDS = {
    "i", "my", "me", "the", "a", "an", "is", "it", "its", "to", "of",
    "and", "or", "in", "on", "at", "for", "with", "this", "that", "you",
    "your", "we", "our", "has", "have", "had", "be", "been", "was", "are",
    "not", "no", "so", "just", "but", "can", "do", "did", "get", "got",
    "will", "would", "could", "should", "what", "why", "how", "when",
    "any", "some", "all", "up", "out", "if", "now", "still", "even",
    "after", "since", "from", "been", "about", "like", "please", "help",
    "hey", "hi", "dear", "thanks", "thank", "apple"
}

def get_top_words(messages, n=10):
    """Get the most common meaningful words across a list of messages."""
    word_counts = Counter()
    for msg in messages:
        words = re.findall(r'\b[a-z]{3,}\b', msg.lower())
        for w in words:
            if w not in STOPWORDS:
                word_counts[w] += 1
    return [word for word, _ in word_counts.most_common(n)]


def get_centroid_examples(messages, umap_vecs, n=8):
    """
    Find messages closest to the centroid of the group.
    The centroid is the average of all UMAP vectors in the group.
    Messages closest to the centroid are most 'representative' of the group.
    """
    if len(messages) == 0:
        return []
    vecs = np.array(umap_vecs)
    centroid = vecs.mean(axis=0)
    # Euclidean distance from each message to centroid
    distances = np.linalg.norm(vecs - centroid, axis=1)
    # Sort by distance, take closest n
    closest_idx = distances.argsort()[:n]
    return [messages[i] for i in closest_idx]


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1: LOADING DATA")
print("=" * 55)

df = pd.read_csv(INPUT_CLUSTERS)
print(f"Cluster assignments: {len(df)} rows")

umap_embeddings = np.load(INPUT_UMAP)
print(f"UMAP embeddings: {umap_embeddings.shape}")

with open(INPUT_MESSAGES) as f:
    messages_list = json.load(f)
thread_id_to_msg = {m["thread_id"]: m["message"] for m in messages_list}

# Build index: cluster_label → list of (message, umap_vec)
cluster_data = defaultdict(list)
for i, row in df.iterrows():
    cluster_data[int(row["cluster_label"])].append({
        "thread_id": row["thread_id"],
        "message":   row["message"],
        "umap_vec":  umap_embeddings[i].tolist(),
    })
print(f"Cluster index built: {len(cluster_data)} clusters")

# Load apple replies for grounding examples
print("Loading apple threads for brand replies...")
with open(INPUT_THREADS) as f:
    all_threads = json.load(f)

thread_id_to_replies = {}
for thread in all_threads:
    replies = [
        t["text"] for t in thread["turns"]
        if t["role"] == "brand" and len(t["text"].strip()) > 20
    ]
    if replies:
        thread_id_to_replies[thread["thread_id"]] = replies

print(f"Apple replies indexed: {len(thread_id_to_replies)} threads")


# ─────────────────────────────────────────────
# STEP 2: BUILD REPORT PER INTENT
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2: BUILDING CLUSTER REPORT")
print("=" * 55)

total_messages = len(df)
report = []

for intent_name, config in MERGE_MAP.items():
    cluster_ids = config["clusters"]

    # Collect all messages and UMAP vectors for this intent
    all_msgs    = []
    all_vecs    = []
    thread_ids  = []

    for cid in cluster_ids:
        for item in cluster_data.get(cid, []):
            all_msgs.append(item["message"])
            all_vecs.append(item["umap_vec"])
            thread_ids.append(item["thread_id"])

    if not all_msgs:
        print(f"  [{intent_name}] — NO MESSAGES FOUND")
        continue

    size = len(all_msgs)
    pct  = size / total_messages * 100

    # Representative examples (centroid-based)
    rep_examples = get_centroid_examples(all_msgs, all_vecs, n=8)

    # Top words
    top_words = get_top_words(all_msgs, n=12)

    # Sample Apple replies from threads in this cluster
    apple_replies = []
    for tid in thread_ids:
        replies = thread_id_to_replies.get(int(tid), [])
        if replies:
            # Take the first substantive reply
            apple_replies.append(replies[0])
        if len(apple_replies) >= 5:
            break

    entry = {
        "intent_name":    intent_name,
        "description":    config["description"],
        "source_clusters": cluster_ids,
        "size":           size,
        "pct_of_total":   round(pct, 1),
        "representative_messages": rep_examples,
        "top_words":      top_words,
        "sample_apple_replies": apple_replies[:5],
    }
    report.append(entry)

    print(f"  [{intent_name}]  {size:>6} messages  ({pct:.1f}%)")

# Sort by size descending
report.sort(key=lambda x: -x["size"])


# ─────────────────────────────────────────────
# STEP 3: SAVE JSON REPORT
# ─────────────────────────────────────────────

with open(OUTPUT_JSON, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {OUTPUT_JSON}")


# ─────────────────────────────────────────────
# STEP 4: SAVE READABLE TEXT REPORT
# ─────────────────────────────────────────────

lines = []
lines.append("=" * 70)
lines.append("CLUSTER REPORT — Apple Support Intent Discovery")
lines.append("=" * 70)
lines.append(f"Total messages: {total_messages:,}")
lines.append(f"Intents:        {len(report)}")
lines.append("")

for entry in report:
    lines.append("=" * 70)
    lines.append(f"INTENT: {entry['intent_name']}")
    lines.append(f"  Size:     {entry['size']:,} messages ({entry['pct_of_total']}%)")
    lines.append(f"  Clusters: {entry['source_clusters']}")
    lines.append("")
    lines.append(f"  Description:")
    lines.append(f"    {entry['description']}")
    lines.append("")
    lines.append(f"  Top words: {', '.join(entry['top_words'])}")
    lines.append("")
    lines.append(f"  Representative messages (closest to centroid):")
    for i, msg in enumerate(entry["representative_messages"], 1):
        lines.append(f"    {i}. {msg[:100]}")
    lines.append("")
    lines.append(f"  Sample Apple replies:")
    for i, reply in enumerate(entry["sample_apple_replies"], 1):
        lines.append(f"    {i}. {reply[:100]}")
    lines.append("")

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved → {OUTPUT_TXT}")


# ─────────────────────────────────────────────
# STEP 5: PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 5: FINAL INTENT SUMMARY")
print("=" * 55)
print(f"\n  {'Intent':<25} {'Messages':>9} {'%':>7}  {'Top words'}")
print(f"  {'─'*70}")

for entry in report:
    words_preview = ", ".join(entry["top_words"][:4])
    bar = "█" * int(entry["pct_of_total"] * 1.2)
    print(
        f"  {entry['intent_name']:<25} "
        f"{entry['size']:>9,} "
        f"({entry['pct_of_total']:>5.1f}%)  "
        f"{words_preview}"
    )

total_classified = sum(e["size"] for e in report)
print(f"\n  Total classified: {total_classified:,} / {total_messages:,}")
print(f"  Noise → general_inquiry: included above")
print()
print("Next: python3 EDA2/05_build_taxonomy.py")