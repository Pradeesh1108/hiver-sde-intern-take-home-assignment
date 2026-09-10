# ============================================================
# EDA2/05_build_taxonomy.py — Build Final Taxonomy Files
# ============================================================
# What this does:
#   Reads cluster_report.json
#   Generates all files the agent needs:
#     - intent_taxonomy.json     (replaces datasets/ version)
#     - few_shot_examples.json   (replaces datasets/ version)
#     - eval_set_to_label.csv    (new 245 examples to label)
#     - retrieval_index.json     (new retrieval pool)
#
#   All outputs saved to EDA2/data/ only.
#   Nothing in datasets/ or agent/ is touched.
#   When ready to swap, copy manually.
#
# Input:  EDA2/data/cluster_report.json
#         EDA2/data/cluster_assignments.csv
#         datasets/apple_threads.json
#
# Output: EDA2/data/intent_taxonomy.json
#         EDA2/data/few_shot_examples.json
#         EDA2/data/eval_set_to_label.csv
#         EDA2/data/retrieval_index.json
#
# Run: python3 EDA2/05_build_taxonomy.py
# ============================================================

import json
import csv
import os
import random
import pandas as pd
from collections import defaultdict

random.seed(42)

INPUT_REPORT    = "EDA2/data/cluster_report.json"
INPUT_CLUSTERS  = "EDA2/data/cluster_assignments.csv"
INPUT_THREADS   = "datasets/apple_threads.json"
OUTPUT_DIR      = "EDA2/data"
OUTPUT_TAXONOMY = "EDA2/data/intent_taxonomy.json"
OUTPUT_FEWSHOT  = "EDA2/data/few_shot_examples.json"
OUTPUT_EVAL     = "EDA2/data/eval_set_to_label.csv"
OUTPUT_RETRIEVAL= "EDA2/data/retrieval_index.json"


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1: LOADING DATA")
print("=" * 55)

with open(INPUT_REPORT) as f:
    report = json.load(f)

df = pd.read_csv(INPUT_CLUSTERS)
print(f"Cluster assignments: {len(df)} rows")

with open(INPUT_THREADS) as f:
    all_threads = json.load(f)

# Build lookup: thread_id → full thread
thread_lookup = {t["thread_id"]: t for t in all_threads}
print(f"Threads loaded: {len(thread_lookup)}")

# Build lookup: thread_id → cluster intent
df["thread_id"] = df["thread_id"].astype(int)
thread_to_intent = dict(zip(df["thread_id"], df["cluster_label"]))

# We need a reverse map: cluster_label → intent_name
# Build from the report's source_clusters
cluster_to_intent = {}
for entry in report:
    for cid in entry["source_clusters"]:
        cluster_to_intent[cid] = entry["intent_name"]

# Assign intent to every row in df
df["intent"] = df["cluster_label"].map(cluster_to_intent).fillna("general_inquiry")

print(f"Intent distribution:")
for intent, count in df["intent"].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {intent:<25} {count:>6} ({pct:.1f}%)")


# ─────────────────────────────────────────────
# STEP 2: BUILD intent_taxonomy.json
# ─────────────────────────────────────────────
# This is the file prompts.py reads to build
# the classification prompt. Contains:
#   - intent name
#   - description (goes into LLM prompt)
#   - keywords (used for retrieval pool filtering)
#   - source_clusters (provenance — which HDBSCAN clusters)
#
# Keywords are derived from the top_words in the cluster report
# rather than hand-crafted. Data-driven keywords.

print("\n" + "=" * 55)
print("STEP 2: BUILDING intent_taxonomy.json")
print("=" * 55)

# Sort report by size descending (same order as intents list)
report_by_name = {e["intent_name"]: e for e in report}

intents = []
intent_names = []

for entry in sorted(report, key=lambda x: -x["size"]):
    name = entry["intent_name"]
    intent_names.append(name)

    intents.append({
        "name":            name,
        "description":     entry["description"],
        "keywords":        entry["top_words"][:8],  # top 8 words as keywords
        "source_clusters": entry["source_clusters"],
        "size":            entry["size"],
        "pct_of_total":    entry["pct_of_total"],
    })

    print(f"  [{name}]")
    print(f"    size={entry['size']:,}  keywords={entry['top_words'][:5]}")

taxonomy = {
    "intents":       intents,
    "intent_names":  intent_names,
    "methodology": (
        "Intents were discovered using unsupervised clustering of 76,656 "
        "customer messages. Pipeline: sentence-transformers/all-mpnet-base-v2 "
        "embeddings → UMAP (768→30 dims) → HDBSCAN (min_cluster_size=100). "
        "106 raw clusters were merged into 10 intents based on support workflow "
        "similarity. Noise points (23.2%) and miscellaneous clusters assigned "
        "to general_inquiry."
    ),
    "labelling_rules": [
        "letter I turns into ? or A? box → autocorrect_i_bug",
        "it → I.T or is → I.S autocorrect → autocorrect_i_bug",
        "battery + iOS update → battery_drain (battery is the primary symptom)",
        "wifi dropping or bluetooth turning on by itself → wifi_bluetooth",
        "phone frozen, black screen, won't turn on → phone_freezing",
        "Apple ID locked, iCloud login, phishing email, iCloud storage → account_access",
        "cracked screen, broken keyboard, burnt charger, Apple Watch hardware → device_hardware",
        "iMessage, App Store, Maps, AirPlay, Siri, alarm, screenshot broken → app_issue",
        "iPhone X order, delivery, billing charge, activation → order_purchase",
        "Apple Pay not available in my country, how does Apple Pay work → apple_pay",
        "iOS/macOS update slow, broken, features missing → ios_update_general",
        "too vague, customer service rant, Genius Bar → general_inquiry (last resort)",
    ],
}

with open(OUTPUT_TAXONOMY, "w") as f:
    json.dump(taxonomy, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {OUTPUT_TAXONOMY}")


# ─────────────────────────────────────────────
# STEP 3: BUILD few_shot_examples.json
# ─────────────────────────────────────────────
# 5 representative examples per intent.
# These go into the classification prompt at runtime.
# We use the centroid-closest examples from the cluster report
# (already computed in 04_inspect.py) — these are the most
# "typical" examples of each intent, not random samples.

print("\n" + "=" * 55)
print("STEP 3: BUILDING few_shot_examples.json")
print("=" * 55)

FEW_SHOT_N = 5
few_shot = {}

for entry in report:
    name     = entry["intent_name"]
    examples = entry["representative_messages"]

    # Take the top 5 centroid-closest examples
    # Filter: at least 5 words, not too long (max 150 chars)
    clean_examples = [
        msg for msg in examples
        if 5 <= len(msg.split()) <= 35
        and len(msg) <= 150
    ][:FEW_SHOT_N]

    # If not enough after filtering, take what we have
    if len(clean_examples) < FEW_SHOT_N:
        clean_examples = examples[:FEW_SHOT_N]

    few_shot[name] = clean_examples
    print(f"  [{name}]  {len(clean_examples)} examples")
    for ex in clean_examples[:2]:
        print(f"    • {ex[:80]}")

with open(OUTPUT_FEWSHOT, "w") as f:
    json.dump({"few_shot_examples": few_shot}, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {OUTPUT_FEWSHOT}")


# ─────────────────────────────────────────────
# STEP 4: BUILD eval_set_to_label.csv
# ─────────────────────────────────────────────
# Stratified sample of 245 examples for human labelling.
# 245 / 10 intents = ~25 per intent (adjusted for size).
#
# Sampling strategy:
#   - Take ~25 per intent
#   - Exclude the few-shot examples (no overlap)
#   - Shuffle so labelling order is random
#   - Save with cluster_hint (which intent the cluster says)
#     so you have a starting point when labelling

print("\n" + "=" * 55)
print("STEP 4: BUILDING eval_set_to_label.csv")
print("=" * 55)

EVAL_SIZE      = 245
PER_INTENT     = EVAL_SIZE // len(intent_names)  # 245/11 = ~22 per intent

# Get few-shot message texts to exclude
fewshot_msgs = set()
for examples in few_shot.values():
    fewshot_msgs.update(examples)

eval_rows = []

for intent_name in intent_names:
    # Get all messages assigned to this intent
    intent_df = df[df["intent"] == intent_name]
    candidates = intent_df[~intent_df["message"].isin(fewshot_msgs)]

    n = min(PER_INTENT, len(candidates))
    sampled = candidates.sample(n=n, random_state=42)

    for _, row in sampled.iterrows():
        eval_rows.append({
            "thread_id":    int(row["thread_id"]),
            "message":      row["message"],
            "cluster_hint": intent_name,  # what clustering says
            "your_label":   "",           # YOU fill this in
        })

    print(f"  [{intent_name}]  sampled {n}/{len(candidates)}")

# Shuffle so not sorted by intent
random.shuffle(eval_rows)

with open(OUTPUT_EVAL, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "thread_id", "message", "cluster_hint", "your_label"
    ])
    writer.writeheader()
    writer.writerows(eval_rows)

print(f"\nSaved {len(eval_rows)} rows → {OUTPUT_EVAL}")
print()
print("HOW TO LABEL:")
print("  1. Open EDA2/data/eval_set_to_label.csv in Google Sheets")
print("  2. Read each 'message'")
print("  3. Glance at 'cluster_hint' — what clustering thinks")
print("  4. Type correct intent in 'your_label'")
print(f"  5. Valid labels: {', '.join(intent_names)}")


# ─────────────────────────────────────────────
# STEP 5: BUILD retrieval_index.json
# ─────────────────────────────────────────────
# The retrieval pool the agent searches when drafting replies.
# Each entry: thread_id, intent, customer_msg, brand_replies
#
# Key difference from old approach:
#   OLD: indexed by keyword_hint (keyword matching)
#   NEW: indexed by intent (cluster-derived)
#
# This means when a message is classified as battery_drain,
# the agent retrieves threads where the clustering ALSO said
# battery_drain — semantically similar, not just keyword-similar.

print("\n" + "=" * 55)
print("STEP 5: BUILDING retrieval_index.json")
print("=" * 55)

# Build lookup: thread_id → intent (from cluster assignment)
tid_to_intent = {}
for _, row in df.iterrows():
    tid_to_intent[int(row["thread_id"])] = row["intent"]

retrieval_index = []
skipped = 0

for thread in all_threads:
    tid = thread["thread_id"]

    # Get intent from cluster assignment
    intent = tid_to_intent.get(tid, "general_inquiry")

    # Get customer opening message
    customer_msg = ""
    for turn in thread["turns"]:
        if turn["role"] == "customer" and turn["text"].strip():
            customer_msg = turn["text"]
            break

    if not customer_msg:
        skipped += 1
        continue

    # Get brand replies
    brand_replies = [
        turn["text"]
        for turn in thread["turns"]
        if turn["role"] == "brand" and len(turn["text"].strip()) > 20
    ]

    if not brand_replies:
        skipped += 1
        continue

    retrieval_index.append({
        "thread_id":     tid,
        "intent":        intent,       # cluster-derived intent
        "customer_msg":  customer_msg,
        "brand_replies": brand_replies,
        "num_turns":     thread["num_turns"],
    })

with open(OUTPUT_RETRIEVAL, "w") as f:
    json.dump(retrieval_index, f, indent=2, ensure_ascii=False)

print(f"Retrieval index: {len(retrieval_index)} entries")
print(f"Skipped: {skipped} (no customer msg or no brand replies)")

# Show breakdown by intent
intent_counts = defaultdict(int)
for entry in retrieval_index:
    intent_counts[entry["intent"]] += 1

print(f"\nRetrieval pool breakdown:")
for intent in intent_names:
    count = intent_counts.get(intent, 0)
    pct   = count / len(retrieval_index) * 100
    print(f"  {intent:<25} {count:>6} ({pct:.1f}%)")

print(f"\nSaved → {OUTPUT_RETRIEVAL}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("DONE — Files generated in EDA2/data/")
print("=" * 55)
print(f"""
  intent_taxonomy.json      10 intents, cluster-derived descriptions
  few_shot_examples.json    5 centroid examples per intent
  eval_set_to_label.csv     {len(eval_rows)} examples for human labelling
  retrieval_index.json      {len(retrieval_index)} threads for reply grounding

These files are in EDA2/data/ — NOT yet in datasets/ or agent/.

To activate the new taxonomy:
  cp EDA2/data/intent_taxonomy.json   datasets/intent_taxonomy.json
  cp EDA2/data/few_shot_examples.json datasets/few_shot_examples.json
  cp EDA2/data/retrieval_index.json   datasets/retrieval_index.json

Then label eval_set_to_label.csv and run:
  python3 -m eval.04_eval --skip-reply --skip-escalate --skip-judge
""")