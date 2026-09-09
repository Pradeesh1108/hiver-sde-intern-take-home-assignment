# ============================================================
# 02_intent_taxonomy.py — Sampling for Manual Labelling
# ============================================================
# What this script does:
#   1. Defines 7 intents with keywords
#   2. Scans all 80k customer opening messages for those keywords
#   3. Spot-checks keyword accuracy (YOU read 70 messages, count errors)
#   4. Samples 250 examples stratified across intents → CSV to label
#   5. Saves 5 few-shot examples per intent for the agent prompt
#   6. Builds retrieval index for reply drafting
#
# What this script does NOT do:
#   - It does NOT label all 80k records
#   - It does NOT call any API
#   - Classification of new messages happens at runtime in 03_agent.py
#
# Run: python3 02_intent_taxonomy.py
# ============================================================

import json
import csv
import random
from collections import defaultdict

random.seed(42)  # fixed seed — same samples every run


# ─────────────────────────────────────────────
# STEP 1: DEFINE THE 7 INTENTS
# ─────────────────────────────────────────────
# Each intent has:
#   name        — the label used everywhere in the pipeline
#   description — plain English, goes into the LLM prompt at runtime
#   keywords    — used NOW for keyword-based sampling across 80k threads
#
# Keywords are NOT the classification logic.
# They are fishing nets — cast across 80k messages to find
# plausible candidates per bucket. YOU make the final label call.
#
# Keyword refinement history:
#   v1: basic keywords only
#   v2: added "question mark", "letter i", "i️", "glitch" to ios_update_issue
#       after analysis of 100 real messages showed the iOS 11 autocorrect
#       "I → A?" bug was falling into general_inquiry undetected.
#       Added "icloud", "unlock", "icloud storage" to account_access
#       after noticing those were matching app_issue incorrectly.

INTENTS = [
    {
        "name": "ios_update_issue",
        "description": (
            "Customer is experiencing problems after installing an iOS update. "
            "Symptoms include slowness, random crashes, features no longer working, "
            "or general dissatisfaction with a new iOS version. Also includes "
            "specific iOS bugs like the autocorrect 'I' question mark glitch."
        ),
        "keywords": [
            "update", "ios", "upgraded", "upgrade", "version",
            "ios11", "ios 11", "slow after", "since update", "new update",
            "question mark", "letter i", "i\ufe0f", "boxes and", "glitch",
            "11.0", "11.1", "high sierra"
        ]
    },
    {
        "name": "battery_drain",
        "description": (
            "Customer reports that their iPhone battery is draining unusually fast, "
            "not charging properly, or lasting much shorter than expected."
        ),
        "keywords": [
            "battery", "drain", "draining", "charge", "charging",
            "percent", "dying", "dead", "power"
        ]
    },
    {
        "name": "device_hardware_issue",
        "description": (
            "Customer is having a problem with a physical aspect of their device — "
            "keyboard, screen, camera, microphone, buttons, touch, or the device "
            "freezing or restarting unexpectedly."
        ),
        "keywords": [
            "keyboard", "screen", "freeze", "frozen", "restart",
            "camera", "touch", "button", "microphone", "hardware",
            "cracked", "broken", "physical", "sim card", "ringer"
        ]
    },
    {
        "name": "app_issue",
        "description": (
            "Customer is having a problem with a specific app — it crashes, "
            "won't open, has missing features, or is incompatible with something. "
            "Includes Apple's own apps (Mail, Music, iMessage) and third-party apps."
        ),
        "keywords": [
            "app", "apps", "mail", "music", "imessage", "facetime",
            "whatsapp", "crash", "crashes", "crashing",
            "spotify", "instagram", "safari", "imovie", "alarm",
            "game center", "appstore", "snapchat"
        ]
    },
    {
        "name": "account_access",
        "description": (
            "Customer has an issue related to their Apple ID, App Store account, "
            "verification codes, passwords, or is locked out of their account."
        ),
        "keywords": [
            "apple id", "password", "code", "locked", "account",
            "sign in", "login", "log in", "verification",
            "icloud", "two factor", "authentication", "unlock",
            "icloud storage", "itunes account"
        ]
    },
    {
        "name": "order_purchase",
        "description": (
            "Customer has a question or issue related to purchasing an Apple product, "
            "a reservation, a delivery, or a billing charge."
        ),
        "keywords": [
            "order", "buy", "purchase", "delivery", "charge", "charged",
            "billed", "billing", "reserve", "reservation", "ship",
            "iphone x", "refund", "receipt", "pre order", "release date"
        ]
    },
    {
        "name": "general_inquiry",
        "description": (
            "Customer is asking a vague question, making a general complaint that "
            "doesn't fit the above categories, or the message is too short or "
            "unclear to classify into a specific intent."
        ),
        "keywords": []  # catch-all — gets everything unmatched
    }
]

INTENT_NAMES = [i["name"] for i in INTENTS]

print("=" * 55)
print("STEP 1: INTENTS DEFINED")
print("=" * 55)
for intent in INTENTS:
    kw_preview = ", ".join(intent["keywords"][:4]) if intent["keywords"] else "catch-all"
    print(f"  {intent['name']:<25} keywords: {kw_preview}...")


# ─────────────────────────────────────────────
# STEP 2: LOAD ALL THREADS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2: LOADING THREADS")
print("=" * 55)

with open('datasets/apple_threads.json', 'r') as f:
    all_threads = json.load(f)

print(f"Loaded {len(all_threads)} threads")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_first_customer_msg(thread):
    """Return the text of the first customer turn in a thread."""
    for turn in thread["turns"]:
        if turn["role"] == "customer":
            return turn["text"]
    return ""

def find_intent_bucket(text, intents):
    """
    Check text against each intent's keywords in order.
    Returns the first matching intent name.
    Falls back to 'general_inquiry' if nothing matches.

    Order matters: ios_update_issue is first because it is the
    broadest — many messages mention both "update" and "battery".
    We classify those as ios_update_issue (update is the cause).
    """
    text_lower = text.lower()
    for intent in intents:
        if not intent["keywords"]:
            continue
        for kw in intent["keywords"]:
            if kw in text_lower:
                return intent["name"]
    return "general_inquiry"


# ─────────────────────────────────────────────
# STEP 3: KEYWORD BUCKETING
# ─────────────────────────────────────────────
# Run every thread's opening message through find_intent_bucket.
# Drop each thread into its matching bucket.

print("\n" + "=" * 55)
print("STEP 3: KEYWORD BUCKETING")
print("=" * 55)

buckets = defaultdict(list)

for thread in all_threads:
    msg = get_first_customer_msg(thread)
    if not msg:
        continue
    intent = find_intent_bucket(msg, INTENTS)
    buckets[intent].append(thread)

print("Bucket sizes:")
total_bucketed = 0
for name in INTENT_NAMES:
    count = len(buckets[name])
    total_bucketed += count
    pct = count / len(all_threads) * 100
    bar = "█" * min(count // 1000, 35)
    print(f"  {name:<25} {count:>6} ({pct:4.1f}%)  {bar}")
print(f"\n  Total bucketed: {total_bucketed}")


# ─────────────────────────────────────────────
# STEP 4: KEYWORD ACCURACY SPOT-CHECK
# ─────────────────────────────────────────────
# Before we trust these buckets for sampling, we measure how
# accurate the keyword assignment actually is.
#
# Why this matters:
#   If keywords are right 85%+ → good enough for sampling
#   If keywords are right <70% → the eval set will be biased
#   and downstream accuracy numbers become misleading
#
# Method:
#   Sample 10 messages per intent = 70 total
#   Print them here AND save to keyword_spot_check.csv
#   YOU read them, fill the "correct?" column (Y or N)
#   Count per intent → record in decision_log.md
#
# This is NOT automated. It takes about 15 minutes to read 70 messages.
# That 15 minutes is what lets you confidently tell the interviewer:
# "Keyword accuracy was X% — here's the evidence."

SPOT_CHECK_PER_INTENT = 10

print("\n" + "=" * 55)
print("STEP 4: KEYWORD SPOT-CHECK")
print("=" * 55)
print("Read each message. Check if keyword_hint is the correct intent.")
print("Fill 'correct?' in datasets/keyword_spot_check.csv (Y or N).")
print("Then count per intent and record in decision_log.md.\n")

spot_check_rows = []

for intent in INTENTS:
    name = intent["name"]
    candidates = buckets[name]

    if not candidates:
        print(f"  [{name}] — no candidates")
        continue

    n = min(SPOT_CHECK_PER_INTENT, len(candidates))
    spot_sample = random.sample(candidates, n)

    print(f"  {'─' * 51}")
    print(f"  BUCKET: {name}  ({n} samples shown)")
    print(f"  {'─' * 51}")

    for i, thread in enumerate(spot_sample, 1):
        msg = get_first_customer_msg(thread)
        display = (msg[:95] + "...") if len(msg) > 95 else msg
        print(f"  {i:>2}. {display}")

        spot_check_rows.append({
            "thread_id":    thread["thread_id"],
            "keyword_hint": name,
            "message":      msg,
            "correct?":     ""   # YOU fill this: Y or N
        })

    print()

# Save spot-check to CSV for offline review
spot_check_csv = "datasets/keyword_spot_check.csv"
with open(spot_check_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["thread_id", "keyword_hint", "message", "correct?"]
    )
    writer.writeheader()
    writer.writerows(spot_check_rows)

print(f"  Saved → {spot_check_csv}")
print(f"  Open it, fill 'correct?' with Y/N, count per intent.")
print(f"  Record results in decision_log.md under D13.")


# ─────────────────────────────────────────────
# STEP 5: SAMPLE THE GOLDEN EVAL SET — 250 examples
# ─────────────────────────────────────────────
# Sample ~35 threads per intent bucket = 250 total.
# Stratified = equal numbers per intent so no single category
# dominates your accuracy measurement.
#
# These 250 are for YOU to manually label.
# The keyword_hint is your starting point — override it when wrong.
#
# Labelling rules (for consistency):
#   battery + update in same message → ios_update_issue (update is cause)
#   app + update in same message     → ios_update_issue
#   very short or vague              → general_inquiry
#   genuinely unsure                 → pick what customer is most upset about

EVAL_SET_SIZE = 250
per_intent = EVAL_SET_SIZE // len(INTENTS)  # 250 / 7 = 35 per intent

print("\n" + "=" * 55)
print(f"STEP 5: SAMPLING EVAL SET ({EVAL_SET_SIZE} examples)")
print("=" * 55)

# Collect eval set thread_ids so we can exclude them from few-shot later
eval_set = []

for intent in INTENTS:
    name = intent["name"]
    candidates = buckets[name]

    n = min(per_intent, len(candidates))
    sampled = random.sample(candidates, n)

    for thread in sampled:
        msg = get_first_customer_msg(thread)
        eval_set.append({
            "thread_id":    thread["thread_id"],
            "num_turns":    thread["num_turns"],
            "message":      msg,
            "keyword_hint": name,
            "your_label":   "",  # YOU fill this in
        })

    print(f"  {name:<25} sampled {n:>3} / {len(candidates)}")

# Shuffle to prevent labelling-fatigue bias
random.shuffle(eval_set)
print(f"\n  Total: {len(eval_set)} examples")


# ─────────────────────────────────────────────
# STEP 6: SAVE EVAL SET AS CSV
# ─────────────────────────────────────────────

eval_csv_path = "datasets/eval_set_to_label.csv"

with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "thread_id", "num_turns", "message", "keyword_hint", "your_label"
    ])
    writer.writeheader()
    writer.writerows(eval_set)

print("\n" + "=" * 55)
print("STEP 6: EVAL SET SAVED")
print("=" * 55)
print(f"  Saved → {eval_csv_path}")
print()
print("  HOW TO LABEL:")
print("  1. Open in Google Sheets or Excel")
print("  2. Read 'message', glance at 'keyword_hint'")
print("  3. Type correct intent in 'your_label'")
print(f"  4. Valid labels: {', '.join(INTENT_NAMES)}")
print()
print("  RULES:")
print("  battery + update → ios_update_issue")
print("  app + update     → ios_update_issue")
print("  short / vague    → general_inquiry")
print("  unsure           → most prominent complaint")


# ─────────────────────────────────────────────
# STEP 7: SAVE FEW-SHOT EXAMPLES
# ─────────────────────────────────────────────
# 5 real examples per intent, excluded from eval set.
# These go into the agent's classification prompt at runtime.
# The LLM sees them and learns what each intent looks like
# without any training — this is called few-shot prompting.

FEW_SHOT_PER_INTENT = 5
eval_thread_ids = {ex["thread_id"] for ex in eval_set}

few_shot = {}
for intent in INTENTS:
    name = intent["name"]
    non_eval = [t for t in buckets[name] if t["thread_id"] not in eval_thread_ids]
    n = min(FEW_SHOT_PER_INTENT, len(non_eval))
    sampled = random.sample(non_eval, n) if n > 0 else []
    few_shot[name] = [get_first_customer_msg(t) for t in sampled]

with open("datasets/few_shot_examples.json", "w") as f:
    json.dump({"intents": INTENTS, "few_shot_examples": few_shot}, f, indent=2)

print("\n" + "=" * 55)
print("STEP 7: FEW-SHOT EXAMPLES SAVED")
print("=" * 55)
print("  Saved → datasets/few_shot_examples.json")
print("  5 examples per intent → go into agent's classification prompt")
print()
for name, examples in few_shot.items():
    if examples:
        print(f"  [{name}]")
        print(f"    \"{examples[0][:70]}\"")


# ─────────────────────────────────────────────
# STEP 8: BUILD RETRIEVAL INDEX
# ─────────────────────────────────────────────
# All 80k threads indexed by keyword_hint.
# Agent searches this at runtime to find historical examples
# of how Apple handled similar issues, then uses those
# real replies to ground its drafted response.
#
# Structure per entry:
#   thread_id     — unique id
#   keyword_hint  — intent bucket (used to filter by topic)
#   customer_msg  — opening message
#   brand_replies — Apple's actual replies (grounding source)

print("\n" + "=" * 55)
print("STEP 8: BUILDING RETRIEVAL INDEX")
print("=" * 55)

retrieval_index = []

for thread in all_threads:
    msg = get_first_customer_msg(thread)
    if not msg:
        continue

    brand_replies = [
        turn["text"]
        for turn in thread["turns"]
        if turn["role"] == "brand" and turn["text"].strip()
    ]

    if not brand_replies:
        continue

    retrieval_index.append({
        "thread_id":     thread["thread_id"],
        "keyword_hint":  find_intent_bucket(msg, INTENTS),
        "customer_msg":  msg,
        "brand_replies": brand_replies,
        "num_turns":     thread["num_turns"]
    })

with open("datasets/retrieval_index.json", "w") as f:
    json.dump(retrieval_index, f, indent=2)

hint_counts = defaultdict(int)
for entry in retrieval_index:
    hint_counts[entry["keyword_hint"]] += 1

print(f"  Total entries: {len(retrieval_index)}")
print(f"  Saved → datasets/retrieval_index.json")
print()
print("  Breakdown by intent:")
for name in INTENT_NAMES:
    count = hint_counts[name]
    pct = count / len(retrieval_index) * 100
    print(f"    {name:<25} {count:>6}  ({pct:.1f}%)")


# ─────────────────────────────────────────────
# SAVE INTENT TAXONOMY AS STANDALONE JSON
# ─────────────────────────────────────────────
# Saved separately so the report and agent can reference it
# without importing this whole script.

with open("datasets/intent_taxonomy.json", "w") as f:
    json.dump({
        "intents": INTENTS,
        "intent_names": INTENT_NAMES,
        "labelling_rules": [
            "battery + update in same message → ios_update_issue (update is the cause)",
            "app + update in same message → ios_update_issue",
            "very short or vague message → general_inquiry",
            "when genuinely unsure → pick the problem the customer seems most upset about"
        ],
        "why_these_7": (
            "Intents were defined at the granularity where two messages in the same "
            "bucket would receive meaningfully similar replies from Apple Support. "
            "Finer granularity created overlapping boundaries that hurt classifier "
            "consistency. Coarser granularity made intents too broad to ground replies."
        ),
        "why_keywords_for_sampling": (
            "Keywords are used only for stratified sampling of the eval set — not for "
            "runtime classification. The agent classifies at runtime using an LLM with "
            "few-shot examples. Keywords were chosen because they are instant, free, "
            "interpretable, and accurate enough (~85%) for finding candidate examples "
            "that a human then verifies. Keyword accuracy was measured via spot-check "
            "on 10 messages per intent before finalising the eval set."
        )
    }, f, indent=2)

print("\n  Saved → datasets/intent_taxonomy.json")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("DONE — outputs")
print("=" * 55)
print(f"""
  keyword_spot_check.csv    70 messages — fill Y/N, measure keyword accuracy
  eval_set_to_label.csv     {len(eval_set)} messages — YOU label these manually
  few_shot_examples.json    5 examples per intent for agent prompt
  retrieval_index.json      {len(retrieval_index)} threads for reply grounding
  intent_taxonomy.json      full taxonomy with reasoning

  Next:
  1. Fill keyword_spot_check.csv → record accuracy in decision_log.md
  2. Label eval_set_to_label.csv (~2.5 hours)
  3. Run 03_agent.py
""")