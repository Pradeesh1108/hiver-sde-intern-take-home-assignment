# ============================================================
# 01_explore.py — Data Exploration & Preparation (v2)
# ============================================================
# Key fix over v1:
#   - Root detection now correctly handles threads whose TRUE root
#     is outside the dataset (parent not in our tweet set)
#   - Thread traversal uses a parent->children map + BFS,
#     not response_tweet_id pointer chasing
#   - Multi-reply (comma-separated response_tweet_id) is handled:
#     we follow ALL children and pick the longest linear branch
#
# Run: python3 01_explore.py
# ============================================================

import pandas as pd
import re
import json
from collections import defaultdict, deque

# ─────────────────────────────────────────────
# STEP 1: LOAD THE DATA
# ─────────────────────────────────────────────

df = pd.read_csv('datasets/Customer Support Twitter Archive/twcs/twcs.csv')   # adjust path to your file

print("=" * 55)
print("STEP 1: RAW DATA OVERVIEW")
print("=" * 55)
print(f"Total rows (tweets): {len(df)}")
print(f"Columns: {df.columns.tolist()}")


# ─────────────────────────────────────────────
# STEP 2: UNDERSTAND inbound
# ─────────────────────────────────────────────

customer_tweets = df[df['inbound'] == True]
brand_tweets    = df[df['inbound'] == False]

print(f"\nCustomer tweets (inbound=True):  {len(customer_tweets)}")
print(f"Brand tweets    (inbound=False): {len(brand_tweets)}")


# ─────────────────────────────────────────────
# STEP 3: FILTER TO AppleSupport
# ─────────────────────────────────────────────
# Strategy: collect ALL tweet_ids that appear in an Apple thread.
# We do this by starting from Apple's replies and walking BOTH directions:
#   - Upward: find parents (in_response_to_tweet_id) of Apple's tweets
#   - Downward: find children (response_tweet_id) of those parents
# We expand outward until no new tweet_ids are found.
# This ensures we capture full multi-turn threads, not just direct pairs.

TARGET_BRAND = 'AppleSupport'

print("\n" + "=" * 55)
print(f"STEP 3: FILTERING TO {TARGET_BRAND}")
print("=" * 55)

# Build fast lookup: tweet_id -> row index (as dict for O(1) access)
# We convert tweet_id to int for consistency (CSV may load as float)
df['tweet_id'] = df['tweet_id'].astype(int)

# Parse in_response_to_tweet_id: single int or NaN
def parse_single_id(val):
    """Convert a single tweet_id field to int, or None if NaN."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

# Parse response_tweet_id: may be comma-separated string of multiple ids
def parse_multi_ids(val):
    """
    Convert response_tweet_id to a list of ints.
    e.g. '119249,119251' -> [119249, 119251]
    e.g. '119249'        -> [119249]
    e.g. NaN             -> []
    """
    if pd.isna(val):
        return []
    return [int(x.strip()) for x in str(val).split(',') if x.strip().isdigit()]

df['parent_id']   = df['in_response_to_tweet_id'].apply(parse_single_id)
df['children_ids'] = df['response_tweet_id'].apply(parse_multi_ids)

# Index: tweet_id -> row as dict (for fast lookup by id)
tweet_by_id = {row['tweet_id']: row.to_dict() for _, row in df.iterrows()}

all_tweet_ids = set(tweet_by_id.keys())

# Seed: Apple's own tweet_ids
apple_tweet_ids = set(df[df['author_id'] == TARGET_BRAND]['tweet_id'].tolist())

# Expand: collect all tweet_ids that are in the same threads as Apple's tweets
# We do BFS: from each Apple tweet, walk up to root and down to leaves
apple_thread_ids = set()

def collect_thread_ids(start_id):
    """
    Given any tweet_id in a thread, collect ALL tweet_ids in that thread.
    Walk UP via parent_id until we leave the dataset,
    then walk DOWN via children_ids until we reach leaves.
    """
    visited = set()
    queue = deque([start_id])

    while queue:
        tid = queue.popleft()
        if tid in visited or tid not in tweet_by_id:
            continue
        visited.add(tid)

        tweet = tweet_by_id[tid]

        # Walk up: add parent if it's in our dataset
        parent = tweet['parent_id']
        if parent and parent in tweet_by_id and parent not in visited:
            queue.append(parent)

        # Walk down: add all children (handles comma-separated multi-replies)
        for child in tweet['children_ids']:
            if child in tweet_by_id and child not in visited:
                queue.append(child)

    return visited

for apple_id in apple_tweet_ids:
    thread_ids = collect_thread_ids(apple_id)
    apple_thread_ids.update(thread_ids)

apple_df = df[df['tweet_id'].isin(apple_thread_ids)].copy()

print(f"Apple-related tweets: {len(apple_df)}")
print(f"  Apple replies:   {len(apple_df[apple_df['inbound'] == False])}")
print(f"  Customer tweets: {len(apple_df[apple_df['inbound'] == True])}")


# ─────────────────────────────────────────────
# STEP 4: CLEAN THE TEXT
# ─────────────────────────────────────────────

def clean_tweet(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'@\S+', '', text)       # remove @mentions
    text = text.replace('&amp;', '&')      # HTML entities
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.replace('&quot;', '"')
    text = re.sub(r'http\S+', '', text)    # remove URLs
    text = re.sub(r'\s+', ' ', text)       # collapse whitespace
    return text.strip()

apple_df['clean_text'] = apple_df['text'].apply(clean_tweet)

print("\n" + "=" * 55)
print("STEP 4: TEXT CLEANING — example")
print("=" * 55)
sample = apple_df[['text', 'clean_text']].head(2)
for _, row in sample.iterrows():
    print(f"  BEFORE: {row['text'][:80]}")
    print(f"  AFTER:  {row['clean_text'][:80]}")
    print()


# ─────────────────────────────────────────────
# STEP 5: BUILD PARENT->CHILDREN MAP (Apple tweets only)
# ─────────────────────────────────────────────
# Now we rebuild the lookup using only Apple-filtered tweets.
# parent_to_children[X] = list of tweet_ids that replied to X

apple_tweet_by_id = {
    row['tweet_id']: row.to_dict()
    for _, row in apple_df.iterrows()
}
apple_id_set = set(apple_tweet_by_id.keys())

# Build children map from in_response_to_tweet_id
# (more reliable than response_tweet_id for ordering —
#  every tweet knows its own parent, but not always all its children)
parent_to_children = defaultdict(list)
for tid, tweet in apple_tweet_by_id.items():
    parent = tweet['parent_id']
    if parent:
        parent_to_children[parent].append(tid)


# ─────────────────────────────────────────────
# STEP 6: FIND LOCAL ROOTS
# ─────────────────────────────────────────────
# A local root is a tweet whose parent is either:
#   a) NaN (no parent at all — true start of conversation), OR
#   b) not in our Apple-filtered set (parent exists but is outside our data)
#
# This fixes the v1 bug where threads starting mid-conversation
# (because their true root wasn't in the dataset) were missed.

local_roots = []
for tid, tweet in apple_tweet_by_id.items():
    parent = tweet['parent_id']
    if parent is None or parent not in apple_id_set:
        local_roots.append(tid)

print("\n" + "=" * 55)
print("STEP 5+6: THREAD STRUCTURE")
print("=" * 55)
print(f"Local root tweets found: {len(local_roots)}")


# ─────────────────────────────────────────────
# STEP 7: RECONSTRUCT THREADS VIA BFS
# ─────────────────────────────────────────────
# From each local root, do a BFS downward through children.
# At each node, if there are multiple children (branching thread),
# we pick the LONGEST branch — the one that leads to the most turns.
# Why longest? Support conversations are linear; branches usually
# happen when multiple users pile onto the same tweet, not when
# the resolution path splits. The longest branch is the main thread.

def get_subtree_depth(tid, children_map, memo={}):
    """
    Recursively compute how many descendants tid has.
    Used to pick the longest branch at a fork.
    """
    if tid in memo:
        return memo[tid]
    children = children_map.get(tid, [])
    if not children:
        memo[tid] = 0
        return 0
    depth = 1 + max(get_subtree_depth(c, children_map, memo) for c in children)
    memo[tid] = depth
    return depth

def reconstruct_thread(root_id, children_map, tweet_lookup, max_depth=15):
    """
    Starting from root_id, follow the conversation downward.
    At each fork (multiple children), take the longest branch.
    Returns an ordered list of tweet dicts (oldest to newest).
    """
    thread = []
    current = root_id
    visited = set()

    for _ in range(max_depth):
        if current is None or current not in tweet_lookup:
            break
        if current in visited:
            break  # cycle guard
        visited.add(current)
        thread.append(tweet_lookup[current])

        children = children_map.get(current, [])
        if not children:
            break  # end of thread

        if len(children) == 1:
            # Linear — no choice needed
            current = children[0]
        else:
            # Fork: pick the child that leads to the longest subtree
            current = max(children, key=lambda c: get_subtree_depth(c, children_map))

    return thread


# ─────────────────────────────────────────────
# STEP 8: BUILD ALL THREADS
# ─────────────────────────────────────────────

def format_thread(raw_thread, brand_name):
    """Convert list of raw tweet dicts into clean structured format.

    Two post-processing steps applied here:

    Step A — Drop empty turns:
        After cleaning, some tweets become empty strings because
        they contained only @mentions and URLs (both stripped).
        A turn with no text carries no information, so we skip it.

    Step B — Merge consecutive same-role turns:
        Sometimes two customer tweets appear back-to-back because
        another user replied to the original customer (piling on),
        or a brand sent a two-part tweet (1/2, 2/2 style).
        In both cases the correct representation is one combined turn,
        not two separate turns of the same role.
        We join their texts with a space.
    """
    turns = []
    for tweet in raw_thread:
        role = 'brand' if tweet['author_id'] == brand_name else 'customer'
        clean = tweet['clean_text']

        # Step A: skip turns that are empty after cleaning
        if not clean.strip():
            continue

        # Step B: if this turn has the same role as the previous one, merge
        if turns and turns[-1]['role'] == role:
            turns[-1]['text'] += ' ' + clean
            turns[-1]['raw_text'] += ' ' + tweet['text']
            # keep tweet_id of the first turn in the merged pair
        else:
            turns.append({
                'role':     role,
                'text':     clean,
                'raw_text': tweet['text'],
                'tweet_id': tweet['tweet_id'],
            })

    # After merging, a thread might be left with only one role — discard it
    roles_present = {t['role'] for t in turns}
    if len(roles_present) < 2 or len(turns) < 2:
        return None  # caller will filter these out

    return {
        'thread_id': turns[0]['tweet_id'],
        'turns':     turns,
        'num_turns': len(turns),
    }

print("\n" + "=" * 55)
print("STEP 7+8: RECONSTRUCTING THREADS")
print("=" * 55)

depth_memo = {}  # shared memo across all threads for efficiency
threads = []

for root_id in local_roots:
    raw_thread = reconstruct_thread(root_id, parent_to_children, apple_tweet_by_id)

    # Only keep threads that have BOTH a customer AND a brand turn
    roles = {t['author_id'] for t in raw_thread}
    has_customer = any(apple_tweet_by_id.get(t['tweet_id'], {}).get('inbound') for t in raw_thread)
    has_brand    = TARGET_BRAND in roles

    if has_customer and has_brand and len(raw_thread) >= 2:
        formatted = format_thread(raw_thread, TARGET_BRAND)
        if formatted is not None:   # format_thread returns None if thread collapses to 1 role
            threads.append(formatted)

print(f"Complete threads (customer + brand, ≥2 turns): {len(threads)}")

# Distribution of thread lengths
from collections import Counter
length_dist = Counter(t['num_turns'] for t in threads)
print("\nThread length distribution:")
for length in sorted(length_dist):
    count = length_dist[length]
    bar = '█' * min(count // 500, 40)
    print(f"  {length:>2} turns: {count:>6}  {bar}")


# ─────────────────────────────────────────────
# STEP 9: SANITY CHECK — print one full thread
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 9: EXAMPLE THREAD (multi-turn)")
print("=" * 55)

# Find a thread with 4+ turns to show reconstruction is working
long_threads = [t for t in threads if t['num_turns'] >= 4]
if long_threads:
    example = long_threads[0]
    print(f"Thread ID: {example['thread_id']} | Turns: {example['num_turns']}")
    for turn in example['turns']:
        print(f"\n  [{turn['role'].upper()}]: {turn['text']}")
else:
    example = threads[0]
    print(f"Thread ID: {example['thread_id']} | Turns: {example['num_turns']}")
    for turn in example['turns']:
        print(f"\n  [{turn['role'].upper()}]: {turn['text']}")


# ─────────────────────────────────────────────
# STEP 10: SAVE
# ─────────────────────────────────────────────

import os
os.makedirs('datasets', exist_ok=True)

output_path = 'datasets/apple_threads.json'
with open(output_path, 'w') as f:
    json.dump(threads, f, indent=2)

print("\n" + "=" * 55)
print("STEP 10: SAVED")
print("=" * 55)
print(f"Saved {len(threads)} threads → {output_path}")
print("\nDone! Run 02_intent_taxonomy.py next.")