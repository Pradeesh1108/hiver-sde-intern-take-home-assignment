# ============================================================
# eval/ml_baseline.py — Standalone ML Baseline
# ============================================================
# Standalone script. No other project files needed.
# Takes the labelled CSV, trains multiple models, reports
# accuracy, precision, recall, F1, and confusion matrix.
#
# Usage:
#   python3 eval/ml_baseline.py
#   python3 eval/ml_baseline.py --csv path/to/your_file.csv
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
from sklearn.svm                     import LinearSVC
from sklearn.naive_bayes             import ComplementNB
from sklearn.ensemble                import RandomForestClassifier
from sklearn.pipeline                import Pipeline
from sklearn.model_selection         import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics                 import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing           import LabelEncoder
from sklearn.base                    import BaseEstimator, TransformerMixin
from sklearn.pipeline                import FeatureUnion


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

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
# TF-IDF + LinearSVC:
#   Support Vector Machine with linear kernel. Often beats
#   logistic regression on text classification tasks because
#   it optimises a different loss function (hinge vs log).
#   Still fast and interpretable.
#
# TF-IDF + ComplementNB:
#   Naive Bayes variant designed for text. Particularly good
#   when classes are imbalanced. Faster than LR/SVC but
#   usually slightly less accurate.
#
# TF-IDF + Random Forest:
#   Ensemble of decision trees. Usually worse than linear
#   models on high-dimensional text (TF-IDF creates thousands
#   of features). Included to demonstrate this point.

# ─────────────────────────────────────────────
# KEYWORD FEATURE AUGMENTATION
# ─────────────────────────────────────────────
# Since domain keywords work better than raw TF-IDF alone,
# we add binary keyword-match flags as extra features.
# The ML model gets explicit domain knowledge to learn from,
# rather than having to rediscover it from 28 training examples.

KEYWORD_MAP = {
    "ios_update_issue":      ["update", "ios", "ios11", "ios 11", "upgrade",
                              "11.0", "11.1", "question mark", "letter i", "glitch"],
    "battery_drain":         ["battery", "drain", "charge", "charging",
                              "percent", "dying", "dead"],
    "device_hardware_issue": ["keyboard", "screen", "freeze", "frozen",
                              "camera", "touch", "broken", "sim card"],
    "app_issue":             ["app", "apps", "mail", "music", "imessage",
                              "crash", "safari", "spotify"],
    "account_access":        ["apple id", "password", "locked", "icloud",
                              "sign in", "verification", "unlock"],
    "order_purchase":        ["order", "buy", "purchase", "delivery",
                              "charge", "reservation", "ship", "refund"],
    "general_inquiry":       [],
}
_INTENT_ORDER = list(KEYWORD_MAP.keys())


class KeywordFeatures(BaseEstimator, TransformerMixin):
    """
    Adds 7 binary features to the feature matrix — one per intent.
    Value is 1.0 if any keyword for that intent matches, else 0.0.
    Concatenated with TF-IDF gives the model domain knowledge directly.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X:
            t = text.lower()
            result.append([
                float(any(kw in t for kw in KEYWORD_MAP[intent]))
                for intent in _INTENT_ORDER
            ])
        return np.array(result)


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
    """Return dict of model_name → sklearn Pipeline."""
    tfidf = build_tfidf_vectorizer

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   LogisticRegression(
                C          = 5.0,   # regularisation — tuned up from default 1.0
                max_iter   = 1000,
                solver     = "lbfgs",
                random_state=42,
            )),
        ]),

        "LinearSVC": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   LinearSVC(
                C        = 1.0,
                max_iter = 2000,
                random_state=42,
            )),
        ]),

        "ComplementNB": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range   = (1, 2),
                max_features  = 10000,
                sublinear_tf  = False,  # NB needs raw counts not log
                min_df        = 2,
                strip_accents = "unicode",
            )),
            ("clf", ComplementNB(alpha=0.1)),
        ]),

        "Random Forest": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   RandomForestClassifier(
                n_estimators = 200,
                max_depth    = None,
                random_state = 42,
                n_jobs       = -1,
            )),
        ]),

        # Keyword-augmented: TF-IDF + explicit keyword signals
        # Combines statistical learning with domain knowledge
        # Expected to beat plain ML but still below keyword baseline
        "ComplementNB + Keywords": Pipeline([
            ("features", FeatureUnion([
                ("tfidf", TfidfVectorizer(
                    ngram_range   = (1, 2),
                    max_features  = 10000,
                    sublinear_tf  = False,
                    min_df        = 2,
                    strip_accents = "unicode",
                )),
                ("keywords", KeywordFeatures()),
            ])),
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
    Only works for Logistic Regression and LinearSVC.
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
    messages_raw, labels = load_csv(args.csv)
    messages = [preprocess(m) for m in messages_raw]
    intent_names = sorted(set(labels))

    print(f"Intents: {intent_names}")
    print(f"Total examples: {len(messages)}")
    print()

    # ── Stratified train/test split ────────────
    # Stratified = every intent appears proportionally in both
    # train and test. Critical with only ~35 examples per intent.
    print(f"Splitting: {int((1-args.test_size)*100)}% train / "
          f"{int(args.test_size*100)}% test (stratified)\n")

    sss = StratifiedShuffleSplit(
        n_splits   = 1,
        test_size  = args.test_size,
        random_state = 42,
    )
    train_idx, test_idx = next(sss.split(messages, labels))

    X_train = [messages[i] for i in train_idx]
    y_train = [labels[i]   for i in train_idx]
    X_test  = [messages[i] for i in test_idx]
    y_test  = [labels[i]   for i in test_idx]

    print(f"Train: {len(X_train)} examples")
    print(f"Test:  {len(X_test)} examples")
    print()

    # ── Cross-validation first ─────────────────
    # CV gives us a more reliable estimate than a single split.
    print(f"{'='*60}")
    print(f"  {args.cv}-FOLD CROSS-VALIDATION (on full dataset)")
    print(f"{'='*60}")
    models    = get_models()
    cv_scores = {}

    for name, pipeline in models.items():
        cv_scores[name] = cross_validate_model(name, pipeline, messages, labels, cv=args.cv)

    # Pick best model by CV score
    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\n  Best model by CV: {best_name} ({cv_scores[best_name]*100:.1f}%)")

    # ── Train/test evaluation ──────────────────
    print(f"\n{'='*60}")
    print(f"  TRAIN/TEST SPLIT RESULTS")
    print(f"{'='*60}")

    all_results = {}
    for name, pipeline in get_models().items():
        result = evaluate_model(
            name, pipeline,
            X_train, y_train,
            X_test, y_test,
            intent_names
        )
        all_results[name] = result
        print_results(result, intent_names)

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
    print(f"  {'Model':<22} {'CV Accuracy':>12} {'Test Accuracy':>14}")
    print(f"  {'─'*50}")
    for name in models:
        cv  = cv_scores[name] * 100
        tst = all_results[name]["accuracy"] * 100
        marker = " ← best" if name == best_name else ""
        print(f"  {name:<22} {cv:>11.1f}% {tst:>13.1f}%{marker}")

    # ── Context: what these numbers mean ───────
    print(f"""
  CONTEXT FOR REPORT:
  ─────────────────────────────────────────────────────
  Random baseline (always guess majority class):
    ~{max(Counter(labels).values())/len(labels)*100:.0f}% (always predicts '{Counter(labels).most_common(1)[0][0]}')

  Keyword baseline:   ~67.8%  (from keyword spot-check)
  Best ML baseline:   {cv_scores[best_name]*100:.1f}%   ({args.cv}-fold CV)
  LLM agent:          TBD     (run eval/04_eval.py)

  Expected order: random < keyword < ML < LLM agent
  ─────────────────────────────────────────────────────
""")

    # ── Save results ───────────────────────────
    import os
    os.makedirs("outputs", exist_ok=True)

    save_data = {
        "cv_scores":   {name: float(score) for name, score in cv_scores.items()},
        "test_scores": {name: float(r["accuracy"]) for name, r in all_results.items()},
        "best_model":  best_name,
        "best_cv_accuracy":   float(cv_scores[best_name]),
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