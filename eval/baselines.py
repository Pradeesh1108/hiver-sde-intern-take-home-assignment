# ============================================================
# eval/baselines.py — Two Baselines for Comparison
# ============================================================
# The assignment requires results vs at least two baselines:
#   Baseline 1 (trivial): keyword matching
#   Baseline 2 (simple):  TF-IDF + Logistic Regression
#
# Why these two?
#   Keyword matching shows the floor — what you get with zero ML.
#   TF-IDF + LR shows what classical ML achieves on this task.
#   Our LLM agent should beat both. If it doesn't, something
#   is wrong with the agent and the report must explain why.
#
# Both baselines are trained/run on the SAME 245 labelled examples
# as the agent eval, so the comparison is fair.
#
# Run: python3 -m eval.baselines
# ============================================================

import json
import csv
import re
from collections import defaultdict


# ─────────────────────────────────────────────
# LOAD SHARED DATA
# ─────────────────────────────────────────────

def _load_intent_taxonomy():
    with open("datasets/intent_taxonomy.json") as f:
        return json.load(f)

def _load_eval_set(path="datasets/eval_set_labelled.csv"):
    """Load the hand-labelled eval set. Returns list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("your_label", "").strip():
                rows.append(row)
    return rows


# ─────────────────────────────────────────────
# BASELINE 1: KEYWORD MATCHING
# ─────────────────────────────────────────────
# The simplest possible classifier:
#   scan the message for keywords, return the first matching intent.
#   Falls back to general_inquiry if nothing matches.
#
# This is identical to the keyword logic in 02_intent_taxonomy.py.
# It is the "trivial baseline" — zero ML, zero training, zero cost.
# Expected accuracy: ~50-65% based on our spot-check results.

class KeywordBaseline:
    """
    Trivial baseline: classify by keyword matching.
    No training required — just scan for keywords.
    """

    def __init__(self):
        taxonomy   = _load_intent_taxonomy()
        self.intents = taxonomy["intents"]

    def predict(self, message: str) -> str:
        """Return the first intent whose keywords appear in the message."""
        text = message.lower()
        for intent in self.intents:
            if not intent["keywords"]:
                continue
            for kw in intent["keywords"]:
                if kw in text:
                    return intent["name"]
        return "general_inquiry"

    def predict_batch(self, messages: list) -> list:
        return [self.predict(m) for m in messages]


# ─────────────────────────────────────────────
# BASELINE 2: TF-IDF + LOGISTIC REGRESSION
# ─────────────────────────────────────────────
# A classical ML pipeline:
#   TF-IDF converts text to a numeric vector
#   Logistic Regression classifies the vector into an intent
#
# TF-IDF = Term Frequency-Inverse Document Frequency
#   "battery" appearing in a battery_drain message gets a high score
#   Common words like "my", "is", "the" get a low score
#   Result: each message becomes a vector of word importance scores
#
# Logistic Regression:
#   Learns a weight for each word-intent pair from training data
#   Fast, interpretable, works well with TF-IDF features
#   No GPU needed — runs on any laptop in seconds
#
# Train/test split: 80/20
#   Train on 196 examples, test on 49
#   Same 245 examples the agent eval uses (different split)
#   We do NOT train on the test portion — that would be cheating

class TFIDFBaseline:
    """
    Simple ML baseline: TF-IDF vectorizer + Logistic Regression.
    Must call .fit() before .predict().
    """

    def __init__(self):
        # Import here so the module loads even without sklearn installed
        # (keyword baseline still works without sklearn)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError:
            raise ImportError(
                "scikit-learn is required for TFIDFBaseline.\n"
                "Install it with: pip install scikit-learn"
            )

        # Pipeline = TF-IDF → Logistic Regression in one object
        # max_features=5000: use the 5000 most informative words
        # C=1.0: regularisation strength (default, works well here)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # unigrams + bigrams
                                     # "battery drain" is more informative
                                     # than "battery" and "drain" separately
                sublinear_tf=True,   # use log(tf) to dampen common words
            )),
            ("clf", LogisticRegression(
                max_iter=1000,       # enough iterations to converge
                C=1.0,
                random_state=42,
            )),
        ])

        self.is_fitted = False

    def fit(self, messages: list, labels: list):
        """
        Train the classifier.

        Args:
            messages: list of message strings (training examples)
            labels:   list of intent name strings (ground truth)
        """
        self.pipeline.fit(messages, labels)
        self.is_fitted = True
        print(f"  [TFIDFBaseline] Trained on {len(messages)} examples")

    def predict(self, message: str) -> str:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return self.pipeline.predict([message])[0]

    def predict_batch(self, messages: list) -> list:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return list(self.pipeline.predict(messages))


# ─────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────

def compute_metrics(predictions: list, ground_truth: list, intent_names: list) -> dict:
    """
    Compute accuracy and per-intent metrics.

    Args:
        predictions:  list of predicted intent strings
        ground_truth: list of true intent strings
        intent_names: list of all valid intent names

    Returns:
        Dict with:
            overall_accuracy: float
            per_intent:       dict of intent → {correct, total, accuracy}
            confusion:        dict of true_label → {predicted_label: count}
    """
    assert len(predictions) == len(ground_truth), "Length mismatch"

    total   = len(predictions)
    correct = sum(p == t for p, t in zip(predictions, ground_truth))

    # Per-intent breakdown
    per_intent = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, true in zip(predictions, ground_truth):
        per_intent[true]["total"]   += 1
        if pred == true:
            per_intent[true]["correct"] += 1

    for name, counts in per_intent.items():
        counts["accuracy"] = (
            counts["correct"] / counts["total"]
            if counts["total"] > 0 else 0.0
        )

    # Confusion matrix as a dict
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, true in zip(predictions, ground_truth):
        confusion[true][pred] += 1

    return {
        "overall_accuracy": correct / total,
        "correct":          correct,
        "total":            total,
        "per_intent":       dict(per_intent),
        "confusion":        {k: dict(v) for k, v in confusion.items()},
    }


def print_metrics(metrics: dict, name: str):
    """Pretty-print metrics for a baseline."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']*100:.1f}%  "
          f"({metrics['correct']}/{metrics['total']})")
    print()
    print("  Per-intent accuracy:")
    for intent, counts in sorted(metrics["per_intent"].items()):
        acc   = counts["accuracy"] * 100
        c     = counts["correct"]
        t     = counts["total"]
        bar   = "█" * int(acc / 10)
        print(f"    {intent:<25} {c:>2}/{t:>2} = {acc:5.1f}%  {bar}")


# ─────────────────────────────────────────────
# MAIN — run both baselines
# python3 -m eval.baselines
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)

    # Load eval set
    print("Loading eval set...")
    eval_set = _load_eval_set("datasets/eval_set_labelled.csv")
    print(f"Loaded {len(eval_set)} labelled examples")

    messages     = [row["message"]    for row in eval_set]
    ground_truth = [row["your_label"] for row in eval_set]

    taxonomy     = _load_intent_taxonomy()
    intent_names = taxonomy["intent_names"]

    # ── Baseline 1: Keyword ──────────────────
    print("\nRunning Baseline 1: Keyword Matching...")
    kb           = KeywordBaseline()
    kb_preds     = kb.predict_batch(messages)
    kb_metrics   = compute_metrics(kb_preds, ground_truth, intent_names)
    print_metrics(kb_metrics, "BASELINE 1: Keyword Matching")

    # ── Baseline 2: TF-IDF + LR ─────────────
    print("\nRunning Baseline 2: TF-IDF + Logistic Regression...")

    # 80/20 train/test split
    indices   = list(range(len(eval_set)))
    random.shuffle(indices)
    split     = int(0.8 * len(indices))
    train_idx = indices[:split]
    test_idx  = indices[split:]

    train_msgs   = [messages[i]     for i in train_idx]
    train_labels = [ground_truth[i] for i in train_idx]
    test_msgs    = [messages[i]     for i in test_idx]
    test_labels  = [ground_truth[i] for i in test_idx]

    tfidf        = TFIDFBaseline()
    tfidf.fit(train_msgs, train_labels)
    tfidf_preds  = tfidf.predict_batch(test_msgs)
    tfidf_metrics = compute_metrics(tfidf_preds, test_labels, intent_names)
    print_metrics(tfidf_metrics, "BASELINE 2: TF-IDF + Logistic Regression (test set)")

    # Save results for the report
    import json, os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_results.json", "w") as f:
        json.dump({
            "keyword_baseline":  kb_metrics,
            "tfidf_baseline":    tfidf_metrics,
            "tfidf_train_size":  len(train_idx),
            "tfidf_test_size":   len(test_idx),
        }, f, indent=2)

    print("\n  Saved → outputs/baseline_results.json")
    print("\nNext: run python3 -m eval.04_eval to evaluate the full agent")