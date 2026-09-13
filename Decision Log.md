# Decision Log — Apple Support Twitter AI Agent

---

**D1 — BFS in both directions for thread reconstruction**
- The naive approach (Apple reply → direct parent only) misses multi-turn threads like `customer → Apple → customer → Apple`
- Used full bidirectional BFS: upward via `in_response_to_tweet_id`, downward via `response_tweet_id`
- Result: 80,491 complete threads vs 74,613 with the one-direction approach

**D2 — Redefined "root" to handle missing parents**
- A root is a tweet whose parent is either `NaN` OR not present in our filtered dataset — not just `NaN`
- The dataset is a snapshot; many threads start mid-conversation because the true root tweet was not captured
- Without this fix, threads with an in-dataset oldest tweet were incorrectly split

**D3 — Merged consecutive same-role turns**
- When two consecutive turns share a role (both customer or both brand), merge their text into one turn
- Causes: another user piling on (`customer → customer → brand`) or Apple's two-part tweets (`1/2` then `2/2`)
- Every downstream component (classifier, drafter, retriever) assumes alternating roles — this enforces it

**D4 — Did not bulk-label 80k threads with an LLM**
- Rejected approach: call the LLM API on all 80k threads to auto-generate training labels
- Reason 1: circular — using an LLM to label data you then use to evaluate the same LLM inflates scores
- Reason 2: expensive — 80k API calls at $0.001 each = $80+, unnecessary when clustering gives free weak labels
- Instead: unsupervised clustering for taxonomy discovery, human labels only for the 242-example eval set

**D5 — Few-shot prompting over a trained ML classifier at runtime**
- A fine-tuned classifier needs thousands of labelled examples — we have 242
- With 7–10 well-defined intents, an LLM with 5 clear examples per intent outperforms a small trained model
- ML classifiers serve as baselines only, not as the production system

**D6 — Replaced hand-crafted keyword taxonomy with unsupervised clustering**
- Keywords defined intents based on assumptions, missing entire categories invisible to surface vocabulary
- Pipeline: `all-mpnet-base-v2` embeddings → UMAP (768→30 dims) → HDBSCAN (min_cluster_size=100)
- Result: 106 raw clusters → merged to 10 intents
- Key discovery: `autocorrect_i_bug` (13,414 messages) was completely invisible to keywords — customers said "question mark", "boxes", "letter I", not "autocorrect"

**D7 — Embedded only the first customer message per thread**
- At runtime the agent classifies the opening message — clustering must match the same conditions
- Later turns contain Apple's clarifying questions ("Which iOS version?") — embedding them pollutes clusters with Apple's language, not the customer's problem
- Threads with fewer than 3 words in the opening message excluded (554 threads, 0.7%)

**D8 — UMAP + HDBSCAN over K-Means**
- K-Means requires specifying K upfront — we do not know how many intents exist in the data
- HDBSCAN discovers K from density and marks noise points (-1) which become `general_inquiry`
- UMAP first: HDBSCAN struggles in 768 dimensions (curse of dimensionality); UMAP to 30 dims fixes this

**D9 — Merged 106 clusters to 10 intents by support workflow similarity**
- Merge rule: two clusters belong to the same intent if Apple's troubleshooting reply would be identical
- Key merges: 14 autocorrect clusters → `autocorrect_i_bug`; 11 iOS/macOS update clusters → `ios_update_general`; 5 connectivity clusters → `wifi_bluetooth`
- Reduced `general_inquiry` from 40.9% to 27.0% by rescuing misplaced clusters

**D10 — apple_pay cluster was Apple Music — merged into app_issue**
- Cluster 20 (4,405 messages) was initially named `apple_pay` based on a quick 3-message peek
- Human labelling of the eval set revealed 21/22 `apple_pay` cluster examples were Apple Music complaints
- Root cause: top words (`music`, `itunes`, `app`, `update`) were the correct signal — we misread the cluster name
- Resolution: merged into `app_issue`, removed `apple_pay` from taxonomy entirely

**D11 — Retrieval indexed by cluster-derived intent, not keywords**
- Old approach: retrieve threads where `keyword_hint` matches — surface vocabulary only
- New approach: retrieve threads where `intent` matches — semantically grouped by clustering
- A message about "battery dying" retrieves threads from the same semantic cluster, not just threads containing the word "battery"

**D12 — Kept emojis and hashtags in text before embedding**
- `all-mpnet-base-v2` was trained on social media text and handles emojis correctly
- Emojis carry real sentiment signal: 😡 in a complaint is semantically different from the same text without it
- Unintended benefit: the `I️` variation selector (U+FE0F) allowed the `autocorrect_i_bug` cluster to form — affected messages literally contain `I️` which clusters differently from plain `I`

**D13 — Parallelised escalator and reply drafter**
- Both steps are LLM API calls, both network-bound
- Escalator only needs `message + intent` — it does not wait for the reply drafter
- `concurrent.futures.ThreadPoolExecutor` runs them in parallel, saving ~1–2 seconds per message

**D14 — Stratified train/test split for ML baseline**
- Random split produced 30.6% accuracy — an artifact of one intent being over-represented in test
- `StratifiedShuffleSplit` guarantees proportional intent representation in both train and test
- Impact: accuracy went from 30.6% (misleading) to 55.1% (fair)

**D15 — Trained ML baseline on 76k cluster-labelled data, not just 193 clean examples**
- Question: does data quantity (76k noisy labels) beat data quality (193 clean labels)?
- Result: ComplementNB on 76k → 65.7% vs ComplementNB on 193 → 43.4% — quantity wins by 22.3pp
- Key insight: the 5.4pp gap between best ML (65.7%) and LLM agent (71.1%) validates clustering quality — if cluster labels were random noise, ML would perform at chance level