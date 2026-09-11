# Run from hiver-task/ directory
# python3 train_on_80k.py

import pandas as pd
import csv
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

print("=" * 55)
print("STEP 1: LOAD 76K CLUSTER-LABELLED TRAINING DATA")
print("=" * 55)

train_df = pd.read_csv("EDA2/data/cluster_assignments.csv")

with open("datasets/intent_taxonomy.json") as f:
    taxonomy = json.load(f)

cluster_to_intent = {}
for intent_info in taxonomy.get("intents", []):
    intent_name = intent_info["name"]
    for cluster in intent_info.get("source_clusters", []):
        cluster_to_intent[cluster] = intent_name

train_df["intent"] = train_df["cluster_label"].map(cluster_to_intent).fillna("general_inquiry")

print(f"Total rows: {len(train_df)}")
print(f"Distribution:")
for intent, count in sorted(Counter(train_df['intent']).items(),
                              key=lambda x: -x[1]):
    print(f"  {intent:<25} {count:>6}")

X_train = train_df["message"].tolist()
y_train = train_df["intent"].tolist()

print(f"\nTraining examples: {len(X_train)}")


print("\n" + "=" * 55)
print("STEP 2: LOAD 242 HUMAN-LABELLED TEST DATA")
print("=" * 55)

X_test, y_test = [], []
with open("datasets/eval_set_labelled.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row.get("your_label", "").strip()
        msg   = row.get("message", "").strip()
        if label and msg:
            X_test.append(msg)
            y_test.append(label)

print(f"Test examples: {len(X_test)}")
print(f"Distribution:")
for intent, count in sorted(Counter(y_test).items(), key=lambda x: -x[1]):
    print(f"  {intent:<25} {count:>3}")


print("\n" + "=" * 55)
print("STEP 3: TRAIN AND EVALUATE")
print("=" * 55)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )),
    ]),
    "ComplementNB": Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=False,
            min_df=2,
        )),
        ("clf", ComplementNB(alpha=0.1)),
    ]),
}

intent_names = sorted(set(y_test))

for name, pipeline in models.items():
    print(f"\n  Training {name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"  Accuracy on 242 human-labelled test set: {acc*100:.1f}%")
    print()
    print(classification_report(
        y_test, y_pred,
        labels=intent_names,
        zero_division=0
    ))


print("=" * 55)
print("COMPARISON")
print("=" * 55)
print()
print("  System                     Accuracy")
print("  ─────────────────────────────────────────")
print("  Random (app_issue)           ~14%")
print("  ComplementNB (193 examples)   43.4%  CV")
print("  Logistic Regression (193)     45.9%  CV")
print("  Keyword matching              57.9%")
print("  ML trained on 76k (noisy)    TBD ← this run")
print("  LLM agent                     71.1%")
print()
print("  Key question: does 76k noisy > 193 clean?")