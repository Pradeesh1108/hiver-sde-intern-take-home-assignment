# Golden Evaluation Set

**File**: `datasets/eval_set_labelled.csv`  
**Size**: 242 examples  
**Intents**: 10  
**Columns**: `thread_id`, `message`, `cluster_hint`, `your_label`

---

## How examples were sampled

The eval set was built from the same 80,491 AppleSupport Twitter threads used to train the system, but sampled and labelled independently of the training pipeline.

**Sampling pipeline:**

1. All 80,491 threads were processed through the EDA clustering pipeline — embeddings, UMAP, HDBSCAN — producing 106 raw clusters merged into 10 named intents. Each thread received a `cluster_hint`: the intent the unsupervised clustering assigned it.

2. From each intent's pool, **~24 messages were randomly sampled** using stratified sampling with `random.seed(42)`. Stratified means every intent is equally represented regardless of its true frequency in the data — intentional, to ensure enough test examples per intent for reliable per-class accuracy measurement.

3. Messages already used as few-shot examples in the classification prompt were **explicitly excluded** from the candidate pool before sampling. Zero overlap between what the agent learns from and what it is evaluated on.

4. The 242 sampled rows were **shuffled** before labelling so they were not sorted by intent. This prevents labelling fatigue and unconscious bias from seeing many consecutive examples of the same class.

**Distribution:**

| Intent | Count | % |
|---|---|---|
| app\_issue | 33 | 13.6% |
| ios\_update\_general | 32 | 13.2% |
| order\_purchase | 30 | 12.4% |
| battery\_drain | 26 | 10.7% |
| autocorrect\_i\_bug | 25 | 10.3% |
| account\_access | 23 | 9.5% |
| wifi\_bluetooth | 21 | 8.7% |
| general\_inquiry | 19 | 7.9% |
| phone\_freezing | 17 | 7.0% |
| device\_hardware | 16 | 6.6% |

---

## How examples were labelled

Each message was labelled manually by reading the raw customer tweet and assigning one of the 10 intent names. No automation was used for `your_label`.

**Labelling process:**

1. Read the `message` column — the raw customer tweet, exactly as it appeared on Twitter.
2. Glanced at `cluster_hint` as a starting point, then independently decided whether to agree or override.
3. Applied these labelling rules consistently:

   - `autocorrect_i_bug` — the letter I turns into a question mark box or weird symbol (`I️`, `A?`). Also covers `it → I.T` and `is → I.S` variants.
   - `battery_drain` — battery dying fast, not charging, dying at high percentage. If battery drain is caused by an update, still label `battery_drain` if battery is the primary complaint.
   - `ios_update_general` — slow, broken, features missing after an iOS or macOS update with no more specific symptom.
   - `wifi_bluetooth` — WiFi dropping, not connecting, Bluetooth turning on by itself.
   - `phone_freezing` — phone frozen, black screen, random restarts, won't turn on.
   - `account_access` — Apple ID locked, iCloud login issues, forgotten password, phishing emails.
   - `device_hardware` — physically broken screen, keyboard, charger, Apple Watch hardware.
   - `app_issue` — specific app not working: iMessage, App Store, Maps, Siri, AirPlay, Apple Music.
   - `order_purchase` — delivery problems, billing charges, iPhone X reservation issues, refund requests.
   - `general_inquiry` — too vague to classify, customer service rant, or genuine how-to question. Last resort only.

**Override rate: 26.4% (64/242)**

The human labeller overrode the cluster hint for 64 of 242 messages. The most common overrides were:

| Cluster said | Human said | Count | Reason |
|---|---|---|---|
| `apple_pay` | `app_issue` | 9 | Cluster 20 was Apple Music, not Apple Pay |
| `apple_pay` | `order_purchase` | 8 | Billing/subscription complaints |
| `general_inquiry` | `ios_update_general` | 6 | Cluster missed update context |
| `general_inquiry` | `autocorrect_i_bug` | 5 | Short messages with I️ emoji |
| `autocorrect_i_bug` | `general_inquiry` | 4 | Message too vague despite I️ emoji |

The `apple_pay` overrides (22 total) revealed a systematic error in the taxonomy: cluster 20 was misnamed during inspection. It contained Apple Music subscription complaints, not Apple Pay availability questions. The cluster was subsequently merged into `app_issue` in the final taxonomy.

---

## What the eval set measures

The eval set is designed to measure **intent classification accuracy** on realistic, noisy customer tweets. It is not a clean benchmark:

- Messages are raw Twitter text — informal, misspelled, with emojis, hashtags, and Unicode variation selectors.
- Some messages are genuinely ambiguous and reasonable labellers would disagree. These ambiguous cases are left in rather than removed — they reflect real-world difficulty.
- The distribution is **stratified, not natural**. In real AppleSupport traffic, `ios_update_general` and `autocorrect_i_bug` are far more common than `order_purchase` or `phone_freezing`. The stratified eval gives equal weight to each intent, which may overstate performance on rare intents and understate performance on common ones.