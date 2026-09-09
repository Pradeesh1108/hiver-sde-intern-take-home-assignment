# ============================================================
# eval/04_eval.py — Full Evaluation Harness
# ============================================================
# Runs the full agent on all 245 labelled examples and measures:
#   1. Intent classification accuracy (vs your ground truth labels)
#   2. Reply quality (via LLM-as-judge in judge.py)
#   3. Escalation rate (what % get escalated and why)
#   4. Baseline comparison (keyword + TF-IDF vs agent)
#
# Outputs:
#   outputs/eval_results.json  — full results for every example
#   outputs/eval_summary.json  — headline numbers for the report
#
# Run: python3 -m eval.04_eval
#
# Expected runtime: ~20-30 minutes for 245 examples
# (two API calls per example: classify + draft_reply
#  one more for judge scoring = three total)
# ============================================================

import json
import csv
import os
import time
from collections import defaultdict

from agent.pipeline  import run as agent_run
from eval.judge      import judge_reply, judge_batch
from eval.baselines  import (
    KeywordBaseline,
    TFIDFBaseline,
    compute_metrics,
    print_metrics,
    _load_intent_taxonomy,
)

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# LOAD EVAL SET
# ─────────────────────────────────────────────

def load_eval_set(path: str = "datasets/eval_set_labelled.csv") -> list:
    """Load the hand-labelled eval CSV. Returns list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("your_label", "").strip()
            msg   = row.get("message", "").strip()
            if label and msg:
                rows.append(row)
    return rows


# ─────────────────────────────────────────────
# STEP 1: RUN AGENT ON ALL EXAMPLES
# ─────────────────────────────────────────────

def run_agent_eval(eval_set: list, verbose: bool = True) -> list:
    """
    Run the full agent pipeline on every example in the eval set.

    For each example we collect:
        predicted_intent  — what the agent classified it as
        true_intent       — your hand label (ground truth)
        reply             — the drafted reply
        decision          — AUTO_HANDLE or ESCALATE
        reason            — escalation reason
        duration_s        — time taken for this example

    Args:
        eval_set: list of dicts from load_eval_set()
        verbose:  print progress

    Returns:
        List of result dicts, one per example.
    """
    results = []
    total   = len(eval_set)

    print(f"\nRunning agent on {total} examples...")
    print("(This will take ~20-30 minutes — 3 API calls per example)\n")

    for i, row in enumerate(eval_set):
        message    = row["message"]
        true_label = row["your_label"]

        # Run the full pipeline
        result = agent_run(message, verbose=False)

        results.append({
            "thread_id":        row.get("thread_id", ""),
            "message":          message,
            "true_intent":      true_label,
            "predicted_intent": result["intent"],
            "correct":          result["intent"] == true_label,
            "reply":            result["reply"],
            "decision":         result["decision"],
            "reason":           result["reason"],
            "confidence":       result["confidence"],
            "duration_s":       result["duration_s"],
            "judge_scores":     {},  # filled in Step 2
        })

        if verbose and (i + 1) % 10 == 0:
            correct_so_far = sum(r["correct"] for r in results)
            acc = correct_so_far / len(results) * 100
            print(f"  [{i+1:>3}/{total}] Running accuracy: {acc:.1f}%")

    return results


# ─────────────────────────────────────────────
# STEP 2: JUDGE REPLY QUALITY
# ─────────────────────────────────────────────

def run_judge_eval(agent_results: list, verbose: bool = True) -> list:
    """
    Score every agent reply using the LLM judge.

    Adds 'judge_scores' to each result dict in-place.

    Args:
        agent_results: list of result dicts from run_agent_eval()
        verbose:       print progress

    Returns:
        The same list with judge_scores filled in.
    """
    print(f"\nJudging {len(agent_results)} replies...")

    for i, result in enumerate(agent_results):
        scores = judge_reply(
            result["message"],
            result["predicted_intent"],
            result["reply"]
        )
        result["judge_scores"] = scores

        if verbose and (i + 1) % 20 == 0:
            print(f"  Judged {i+1}/{len(agent_results)}...")

    return agent_results


# ─────────────────────────────────────────────
# STEP 3: RUN BASELINES
# ─────────────────────────────────────────────

def run_baseline_eval(eval_set: list) -> dict:
    """
    Run both baselines on the eval set and return their metrics.

    Args:
        eval_set: list of dicts from load_eval_set()

    Returns:
        Dict with keyword_metrics and tfidf_metrics.
    """
    import random
    random.seed(42)

    taxonomy     = _load_intent_taxonomy()
    intent_names = taxonomy["intent_names"]
    messages     = [row["message"]    for row in eval_set]
    ground_truth = [row["your_label"] for row in eval_set]

    # Baseline 1: keyword
    print("\nRunning Baseline 1: Keyword Matching...")
    kb       = KeywordBaseline()
    kb_preds = kb.predict_batch(messages)
    kb_metrics = compute_metrics(kb_preds, ground_truth, intent_names)
    print_metrics(kb_metrics, "Keyword Baseline")

    # Baseline 2: TF-IDF — 80/20 split
    print("\nRunning Baseline 2: TF-IDF + Logistic Regression...")
    indices   = list(range(len(eval_set)))
    random.shuffle(indices)
    split     = int(0.8 * len(indices))
    train_idx = indices[:split]
    test_idx  = indices[split:]

    train_msgs    = [messages[i]     for i in train_idx]
    train_labels  = [ground_truth[i] for i in train_idx]
    test_msgs     = [messages[i]     for i in test_idx]
    test_labels   = [ground_truth[i] for i in test_idx]

    tfidf = TFIDFBaseline()
    tfidf.fit(train_msgs, train_labels)
    tfidf_preds   = tfidf.predict_batch(test_msgs)
    tfidf_metrics = compute_metrics(tfidf_preds, test_labels, intent_names)
    print_metrics(tfidf_metrics, "TF-IDF + LR Baseline (test set)")

    return {
        "keyword":        kb_metrics,
        "tfidf":          tfidf_metrics,
        "tfidf_train_n":  len(train_idx),
        "tfidf_test_n":   len(test_idx),
    }


# ─────────────────────────────────────────────
# STEP 4: COMPUTE SUMMARY STATISTICS
# ─────────────────────────────────────────────

def compute_summary(agent_results: list, baseline_results: dict) -> dict:
    """
    Compute headline numbers for the report.

    Returns:
        Dict with all the key metrics in one place.
    """
    taxonomy     = _load_intent_taxonomy()
    intent_names = taxonomy["intent_names"]

    predictions  = [r["predicted_intent"] for r in agent_results]
    ground_truth = [r["true_intent"]       for r in agent_results]

    # Classification accuracy
    clf_metrics = compute_metrics(predictions, ground_truth, intent_names)

    # Reply quality — average judge scores
    valid_scores = [
        r["judge_scores"] for r in agent_results
        if r["judge_scores"] and not r["judge_scores"].get("error")
    ]

    def avg(key):
        vals = [s[key] for s in valid_scores if s.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    avg_scores = {
        "relevance":     avg("relevance"),
        "tone":          avg("tone"),
        "actionability": avg("actionability"),
        "conciseness":   avg("conciseness"),
        "overall":       avg("overall"),
        "n_scored":      len(valid_scores),
    }

    # Escalation stats
    escalated = [r for r in agent_results if r["decision"] == "ESCALATE"]
    escalation_rate = len(escalated) / len(agent_results)

    # Escalation breakdown by intent
    esc_by_intent = defaultdict(int)
    total_by_intent = defaultdict(int)
    for r in agent_results:
        total_by_intent[r["true_intent"]] += 1
        if r["decision"] == "ESCALATE":
            esc_by_intent[r["true_intent"]] += 1

    # Average duration
    avg_duration = sum(r["duration_s"] for r in agent_results) / len(agent_results)

    return {
        "classification": {
            "overall_accuracy": clf_metrics["overall_accuracy"],
            "correct":          clf_metrics["correct"],
            "total":            clf_metrics["total"],
            "per_intent":       clf_metrics["per_intent"],
        },
        "reply_quality":   avg_scores,
        "escalation": {
            "rate":        escalation_rate,
            "total":       len(escalated),
            "by_intent":   dict(esc_by_intent),
        },
        "baselines": {
            "keyword_accuracy": baseline_results["keyword"]["overall_accuracy"],
            "tfidf_accuracy":   baseline_results["tfidf"]["overall_accuracy"],
            "agent_accuracy":   clf_metrics["overall_accuracy"],
        },
        "performance": {
            "avg_duration_s": round(avg_duration, 2),
            "total_examples": len(agent_results),
        }
    }


def print_summary(summary: dict):
    """Print the headline numbers clearly."""
    clf  = summary["classification"]
    rq   = summary["reply_quality"]
    esc  = summary["escalation"]
    base = summary["baselines"]

    print(f"\n{'='*55}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*55}")

    print(f"\n  CLASSIFICATION ACCURACY")
    print(f"  Agent:    {base['agent_accuracy']*100:.1f}%")
    print(f"  TF-IDF:   {base['tfidf_accuracy']*100:.1f}%  (simple baseline)")
    print(f"  Keyword:  {base['keyword_accuracy']*100:.1f}%  (trivial baseline)")

    print(f"\n  REPLY QUALITY (judge scores, 1-5)")
    print(f"  Relevance:     {rq.get('relevance', 'N/A')}")
    print(f"  Tone:          {rq.get('tone', 'N/A')}")
    print(f"  Actionability: {rq.get('actionability', 'N/A')}")
    print(f"  Conciseness:   {rq.get('conciseness', 'N/A')}")
    print(f"  Overall:       {rq.get('overall', 'N/A')}")
    print(f"  (n={rq.get('n_scored')} replies scored)")

    print(f"\n  ESCALATION")
    print(f"  Rate:     {esc['rate']*100:.1f}% ({esc['total']}/{clf['total']} escalated)")

    print(f"\n  PER-INTENT ACCURACY")
    for intent, counts in sorted(clf["per_intent"].items()):
        acc = counts["accuracy"] * 100
        c   = counts["correct"]
        t   = counts["total"]
        bar = "█" * int(acc / 10)
        print(f"    {intent:<25} {c:>2}/{t:>2} = {acc:5.1f}%  {bar}")


# ─────────────────────────────────────────────
# MAIN
# python3 -m eval.04_eval
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM judge scoring (faster, classification metrics only)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate first N examples (for quick testing)"
    )
    args = parser.parse_args()

    # Load eval set
    print("Loading eval set...")
    eval_set = load_eval_set("datasets/eval_set_labelled.csv")
    print(f"Loaded {len(eval_set)} labelled examples")

    if args.limit:
        eval_set = eval_set[:args.limit]
        print(f"Limited to first {args.limit} examples")

    # Step 1: Run agent
    agent_results = run_agent_eval(eval_set, verbose=True)

    # Step 2: Judge replies (optional — slow)
    if not args.skip_judge:
        agent_results = run_judge_eval(agent_results, verbose=True)
    else:
        print("\n  Skipping judge scoring (--skip-judge flag set)")

    # Step 3: Run baselines
    baseline_results = run_baseline_eval(eval_set)

    # Step 4: Summary
    summary = compute_summary(agent_results, baseline_results)
    print_summary(summary)

    # Save everything
    output = {
        "agent_results":    agent_results,
        "baseline_results": baseline_results,
        "summary":          summary,
    }

    with open("outputs/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    with open("outputs/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved → outputs/eval_results.json")
    print(f"  Saved → outputs/eval_summary.json")
    print(f"\n  Next: python3 -m eval.judge (measure human-judge agreement)")