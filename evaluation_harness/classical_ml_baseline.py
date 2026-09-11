# ============================================================
# evaluation_harness/classical_ml_baseline.py — Standalone ML Baseline
# ============================================================
# Standalone script. No other project files needed.
# Takes the labelled CSV, trains multiple models, reports
# accuracy, precision, recall, F1, and confusion matrix.
#
# Usage:
#   python3 evaluation_harness/classical_ml_baseline.py
#   python3 evaluation_harness/classical_ml_baseline.py --csv path/to/your_file.csv
#
# Install dependencies:
#   pip install scikit-learn pandas numpy
# ============================================================

import argparse
import json
import csv
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.naive_bayes             import ComplementNB
from sklearn.pipeline                import Pipeline
from sklearn.model_selection         import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics                 import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing           import LabelEncoder


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

def load_76k_training_data(csv_path: str = "EDA2/data/cluster_assignments.csv", taxonomy_path: str = "datasets/intent_taxonomy.json") -> tuple:
    import pandas as pd
    
    train_df = pd.read_csv(csv_path)

    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    cluster_to_intent = {}
    for intent_info in taxonomy.get("intents", []):
        intent_name = intent_info["name"]
        for cluster in intent_info.get("source_clusters", []):
            cluster_to_intent[cluster] = intent_name

    train_df["intent"] = train_df["cluster_label"].map(cluster_to_intent).fillna("general_inquiry")
    
    messages = train_df["message"].tolist()
    labels = train_df["intent"].tolist()
    
    print(f"Loaded {len(messages)} training examples from {csv_path}")
    return messages, labels



def load_csv(path: str) -> tuple:
    """
    Load the labelled eval CSV.

    Expected columns: thread_id, num_turns, message, keyword_hint, your_label
    Returns (messages, labels) as lists of strings.
    Skips rows where your_label is empty.
    """
    messages = []
    labels   = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label   = row.get("your_label", "").strip()
            message = row.get("message", "").strip()
            if label and message:
                messages.append(message)
                labels.append(label)

    print(f"Loaded {len(messages)} labelled examples from {path}")
    print()

    # Show distribution
    dist = Counter(labels)
    print("Label distribution:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {label:<25} {count:>3}  {bar}")
    print()

    return messages, labels


# ─────────────────────────────────────────────
# STEP 2: TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Light preprocessing before TF-IDF.

    We keep this minimal — TF-IDF handles most of it.
    We just lowercase and remove obvious noise.

    We do NOT remove stopwords here because in short tweets,
    every word can be meaningful — "not working" loses meaning
    if you remove "not".
    """
    import re
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\S+", "", text)
    # Remove special unicode characters (emojis etc)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# STEP 3: BUILD MODELS TO COMPARE
# ─────────────────────────────────────────────
# We try 4 models on the same data.
# Each is a sklearn Pipeline: TF-IDF vectorizer → classifier.
# Using a Pipeline means the vectorizer is fit only on training
# data — never on test data. This prevents data leakage.
#
# Why these 4?
#
# TF-IDF + Logistic Regression:
#   The standard baseline. Fast, interpretable, works well
#   on short text. Can inspect feature weights per class.
#


def build_tfidf_vectorizer():
    """
    Build the TF-IDF vectorizer with settings tuned for short tweets.

    Key settings explained:
      ngram_range=(1,2):  learn unigrams AND bigrams
                          "battery drain" is more specific than "battery" alone
      max_features=10000: keep the 10k most informative terms
                          avoids overfitting on rare words
      sublinear_tf=True:  use log(1+tf) instead of raw term frequency
                          dampens effect of very frequent words
      min_df=2:           ignore terms that appear in fewer than 2 documents
                          removes typos and rare noise
      strip_accents='unicode': normalise accented characters
    """
    return TfidfVectorizer(
        ngram_range    = (1, 2),
        max_features   = 10000,
        sublinear_tf   = True,
        min_df         = 2,
        strip_accents  = "unicode",
        analyzer       = "word",
    )


def get_models() -> dict:
    """
    Return the two ML models used as baselines.

    Why these two:
      Logistic Regression — the standard text classification baseline.
        Linear model, learns a weight per word per intent.
        Interpretable: you can inspect which words drive each prediction.
        C=5.0 (slightly higher than default 1.0) gives less regularisation,
        which helps on our small dataset where overfitting is less of a risk
        than underfitting.

      ComplementNB — Naive Bayes variant designed specifically for text.
        Trains each class on the COMPLEMENT of its data (all other classes),
        which makes it more robust when classes are imbalanced.
        Does NOT use log(tf) — raw counts work better for NB math.
        alpha=0.1 is lighter smoothing than default 1.0, better for short text.
        Consistently outperformed plain Logistic Regression on our data (51% vs 47%).

    Both use TF-IDF with bigrams (ngram_range=(1,2)) so phrases like
    "battery drain" and "ios update" are treated as single features.
    """
    tfidf = build_tfidf_vectorizer

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   LogisticRegression(
                C           = 5.0,
                max_iter    = 1000,
                solver      = "lbfgs",
                random_state= 42,
            )),
        ]),

        "ComplementNB": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range   = (1, 2),
                max_features  = 10000,
                sublinear_tf  = False,   # NB uses raw counts, not log(tf)
                min_df        = 2,
                strip_accents = "unicode",
            )),
            ("clf", ComplementNB(alpha=0.1)),
        ]),
    }


# ─────────────────────────────────────────────
# STEP 4: EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(name: str, pipeline, X_train, y_train, X_test, y_test,
                   intent_names: list) -> dict:
    """
    Train the model and compute all metrics on the test set.

    Returns dict with accuracy, per-class metrics, confusion matrix.
    """
    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels       = intent_names,
        output_dict  = True,
        zero_division= 0,
    )
    cm     = confusion_matrix(y_test, y_pred, labels=intent_names)

    return {
        "name":       name,
        "accuracy":   acc,
        "report":     report,
        "confusion":  cm,
        "y_pred":     y_pred,
        "pipeline":   pipeline,
    }


def cross_validate_model(name: str, pipeline, X, y, cv: int = 5) -> float:
    """
    Run stratified k-fold cross-validation.

    Why cross-validation?
    With only 245 examples, a single 80/20 split is noisy —
    the exact accuracy depends on which 49 examples ended up in
    the test set. Cross-validation runs 5 different splits and
    averages the results, giving a more stable estimate.

    Returns mean accuracy across all folds.
    """
    scores = cross_val_score(
        pipeline, X, y,
        cv      = cv,
        scoring = "accuracy",
        n_jobs  = -1,
    )
    print(f"  {name:<22} CV scores: {[f'{s:.3f}' for s in scores]}  "
          f"mean={scores.mean():.3f}  std=±{scores.std():.3f}")
    return scores.mean()


# ─────────────────────────────────────────────
# STEP 5: PRINT RESULTS
# ─────────────────────────────────────────────

def print_results(result: dict, intent_names: list):
    """Print full metrics for one model."""
    print(f"\n{'='*60}")
    print(f"  {result['name']}")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {result['accuracy']*100:.1f}%  "
          f"({int(result['accuracy']*len(result['y_pred']))}/{len(result['y_pred'])})")
    print()

    # Per-intent table
    print(f"  {'Intent':<25} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
    print(f"  {'─'*60}")
    report = result["report"]
    for intent in intent_names:
        if intent not in report:
            continue
        r = report[intent]
        print(f"  {intent:<25} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1-score']:>6.3f} {int(r['support']):>8}")

    # Macro averages
    macro = report.get("macro avg", {})
    print(f"  {'─'*60}")
    print(f"  {'MACRO AVG':<25} {macro.get('precision',0):>10.3f} "
          f"{macro.get('recall',0):>8.3f} {macro.get('f1-score',0):>6.3f}")


def print_confusion_matrix(result: dict, intent_names: list):
    """Print confusion matrix with short intent labels."""
    print(f"\n  Confusion Matrix — {result['name']}")
    print(f"  (rows=true, cols=predicted)\n")

    # Shorten intent names for display
    short = {
        "ios_update_issue":      "ios_upd",
        "battery_drain":         "battery",
        "device_hardware_issue": "hw_issue",
        "app_issue":             "app",
        "account_access":        "account",
        "order_purchase":        "order",
        "general_inquiry":       "general",
    }

    labels = [short.get(n, n[:8]) for n in intent_names]
    cm     = result["confusion"]

    # Header
    header = "              " + "  ".join(f"{l:>8}" for l in labels)
    print(f"  {header}")

    for i, row_label in enumerate(labels):
        row = "  ".join(
            f"{'['+str(cm[i][j])+']':>8}" if cm[i][j] > 0 else f"{'·':>8}"
            for j in range(len(labels))
        )
        print(f"  {row_label:>12}  {row}")


def print_top_features(pipeline, intent_names: list, n: int = 8):
    """
    Print the top N TF-IDF features (words) per intent.
    Only works for Logistic Regression and ComplementNB (if it has coef_).
    Helps verify the model learned meaningful signals.
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        clf        = pipeline.named_steps["clf"]
        features   = vectorizer.get_feature_names_out()

        if not hasattr(clf, "coef_"):
            return

        print(f"\n  Top {n} words per intent:")
        for i, intent in enumerate(intent_names):
            if i >= len(clf.coef_):
                break
            top_idx  = clf.coef_[i].argsort()[-n:][::-1]
            top_words = [features[j] for j in top_idx]
            print(f"  {intent:<25} {', '.join(top_words)}")

    except Exception:
        pass   # skip silently if model doesn't support this


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone ML baseline for Apple Support intent classification"
    )
    parser.add_argument(
        "--csv",
        default="datasets/eval_set_labelled.csv",
        help="Path to the labelled CSV file (default: datasets/eval_set_labelled.csv)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--save",
        default="outputs/ml_baseline_results.json",
        help="Where to save results JSON"
    )
    args = parser.parse_args()

    # ── Load ───────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 1: LOAD 76K CLUSTER-LABELLED TRAINING DATA")
    print("=" * 55)
    X_train_raw, y_train = load_76k_training_data()
    X_train = [preprocess(m) for m in X_train_raw]

    print("\n" + "=" * 55)
    print("STEP 2: LOAD 242 HUMAN-LABELLED TEST DATA")
    print("=" * 55)
    X_test_raw, y_test = load_csv(args.csv)
    X_test = [preprocess(m) for m in X_test_raw]

    intent_names = sorted(set(y_test))

    print(f"Train: {len(X_train)} examples")
    print(f"Test:  {len(X_test)} examples")
    print()

    # ── Train/test evaluation ──────────────────
    print(f"\n{'='*60}")
    print(f"  TRAIN/TEST SPLIT RESULTS")
    print(f"{'='*60}")

    all_results = {}
    models = get_models()
    for name, pipeline in models.items():
        result = evaluate_model(
            name, pipeline,
            X_train, y_train,
            X_test, y_test,
            intent_names
        )
        all_results[name] = result
        print_results(result, intent_names)
        
    # Find best model
    best_name = max(all_results, key=lambda k: all_results[k]["accuracy"])

    # ── Best model deep dive ───────────────────
    best_result = all_results[best_name]

    print(f"\n{'='*60}")
    print(f"  BEST MODEL DEEP DIVE: {best_name}")
    print(f"{'='*60}")
    print_confusion_matrix(best_result, intent_names)
    print_top_features(best_result["pipeline"], intent_names)

    # ── Comparison table ───────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'Test Accuracy':>14}")
    print(f"  {'─'*37}")
    for name in models:
        tst = all_results[name]["accuracy"] * 100
        marker = " ← best" if name == best_name else ""
        print(f"  {name:<22} {tst:>13.1f}%{marker}")

    # ── Context: what these numbers mean ───────
    print(f"""
  CONTEXT FOR REPORT:
  ─────────────────────────────────────────────────────
  Random baseline (always guess majority class):
    ~{max(Counter(y_test).values())/len(y_test)*100:.0f}% (always predicts '{Counter(y_test).most_common(1)[0][0]}')

  Keyword baseline:   ~67.8%  (from keyword spot-check)
  Best ML baseline:   {all_results[best_name]['accuracy']*100:.1f}%   (trained on 76k)
  LLM agent:          TBD     (run evaluation_harness/automated_metrics.py)

  Expected order: random < keyword < ML < LLM agent
  ─────────────────────────────────────────────────────
""")

    # ── Save results ───────────────────────────
    import os
    os.makedirs("outputs", exist_ok=True)

    save_data = {
        "test_scores": {name: float(r["accuracy"]) for name, r in all_results.items()},
        "best_model":  best_name,
        "best_test_accuracy": float(all_results[best_name]["accuracy"]),
        "per_intent_report":  {
            name: {
                intent: result["report"].get(intent, {})
                for intent in intent_names
            }
            for name, result in all_results.items()
        },
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "n_intents":  len(intent_names),
        "intent_names": intent_names,
    }

    with open(args.save, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"  Results saved → {args.save}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()