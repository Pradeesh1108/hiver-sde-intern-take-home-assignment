# ============================================================
# evaluation_harness/automated_metrics.py — Full Evaluation Harness
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
# Run: python3 -m evaluation_harness.automated_metrics
# ============================================================

import json
import csv
import os
import argparse
import time
from collections import defaultdict

from agent.pipeline  import run as agent_run
from evaluation_harness.llm_as_judge      import judge_reply, judge_batch
from evaluation_harness.classical_ml_baseline import get_models, preprocess, load_76k_training_data
from sklearn.model_selection import StratifiedShuffleSplit

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# EVAL HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _load_intent_taxonomy():
    with open("datasets/intent_taxonomy.json") as f:
        return json.load(f)

def compute_metrics(predictions: list, ground_truth: list, intent_names: list) -> dict:
    assert len(predictions) == len(ground_truth), "Length mismatch"
    total   = len(predictions)
    correct = sum(p == t for p, t in zip(predictions, ground_truth))

    per_intent = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, true in zip(predictions, ground_truth):
        per_intent[true]["total"]   += 1
        if pred == true:
            per_intent[true]["correct"] += 1

    for name, counts in per_intent.items():
        counts["accuracy"] = (counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0)

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
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']*100:.1f}%  "
          f"({metrics['correct']}/{metrics['total']})")
    print("\n  Per-intent accuracy:")
    for intent, counts in sorted(metrics["per_intent"].items()):
        acc = counts["accuracy"] * 100
        bar = "█" * int(acc / 10)
        print(f"    {intent:<25} {counts['correct']:>2}/{counts['total']:>2} = {acc:5.1f}%  {bar}")

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

def run_agent_eval(eval_set: list, verbose: bool = True, skip_reply: bool = False, skip_escalate: bool = False) -> list:
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
    total   = len(eval_set)
    results = [None] * total
    
    import os, json, threading
    checkpoint_file = "outputs/checkpoint_agent.jsonl"
    cached_results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        if "thread_id" in res:
                            cached_results[res["thread_id"]] = res
                    except Exception:
                        pass
    
    print(f"\nRunning agent on {total} examples...")
    if cached_results:
        print(f"(Resuming from checkpoint: {len(cached_results)} already completed)")

    lock = threading.Lock()

    def process_row(i, row):
        tid = row.get("thread_id", "")
        if tid and tid in cached_results:
            return i, cached_results[tid]

        import time
        message    = row["message"]
        true_label = row["your_label"]
        result = agent_run(message, verbose=False, skip_reply=skip_reply, skip_escalate=skip_escalate)
        
        if os.getenv("LLM_PROVIDER", "groq").lower() != "ollama":
            time.sleep(6.5)
            
        final_res = {
            "thread_id":        tid,
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
        }
        
        with lock:
            with open(checkpoint_file, "a") as f:
                f.write(json.dumps(final_res) + "\n")
                
        return i, final_res

    import os
    if os.getenv("LLM_PROVIDER", "groq").lower() == "ollama":
        import concurrent.futures
        max_workers = int(os.getenv("MAX_WORKERS", 8))
        print(f"  (Running in parallel using ThreadPoolExecutor with {max_workers} workers for Ollama)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_row, i, row) for i, row in enumerate(eval_set)]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idx, res = future.result()
                results[idx] = res
                completed += 1
                if verbose and completed % 1 == 0:
                    correct_so_far = sum(r["correct"] for r in results if r is not None)
                    acc = correct_so_far / completed * 100
                    print(f"  [{completed:>3}/{total}] Running accuracy: {acc:.1f}%")
    else:
        for i, row in enumerate(eval_set):
            idx, res = process_row(i, row)
            results[idx] = res
            if verbose and (i + 1) % 1 == 0:
                correct_so_far = sum(r["correct"] for r in results[:i+1])
                acc = correct_so_far / (i + 1) * 100
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
    import os, json, threading
    checkpoint_file = "outputs/checkpoint_judge.jsonl"
    cached_scores = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "thread_id" in data:
                            cached_scores[data["thread_id"]] = data["scores"]
                    except Exception:
                        pass
                        
    print(f"\nJudging {len(agent_results)} replies...")
    if cached_scores:
        print(f"(Resuming from checkpoint: {len(cached_scores)} already judged)")

    lock = threading.Lock()

    def process_judge(i, result):
        tid = result.get("thread_id", "")
        if tid and tid in cached_scores:
            return i, cached_scores[tid]

        scores = judge_reply(
            result["message"],
            result["predicted_intent"],
            result["reply"]
        )
        
        import os, time
        if os.getenv("LLM_PROVIDER", "groq").lower() != "ollama":
            time.sleep(3)
            
        with lock:
            with open(checkpoint_file, "a") as f:
                f.write(json.dumps({"thread_id": tid, "scores": scores}) + "\n")
                
        return i, scores

    import os
    if os.getenv("LLM_PROVIDER", "groq").lower() == "ollama":
        import concurrent.futures
        max_workers = int(os.getenv("MAX_WORKERS", 8))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_judge, i, res) for i, res in enumerate(agent_results)]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idx, scores = future.result()
                agent_results[idx]["judge_scores"] = scores
                completed += 1
                if verbose and completed % 1 == 0:
                    print(f"  Judged {completed}/{len(agent_results)}...")
    else:
        for i, result in enumerate(agent_results):
            idx, scores = process_judge(i, result)
            agent_results[idx]["judge_scores"] = scores
            if verbose and (i + 1) % 1 == 0:
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

    # Prepare test data (the 242 human-labeled examples)
    test_msgs     = [preprocess(row["message"]) for row in eval_set]
    test_labels   = [row["your_label"] for row in eval_set]

    print("\nRunning ML Baselines...")
    print("  Loading 76k training dataset...")
    train_msgs_raw, train_labels = load_76k_training_data()
    train_msgs = [preprocess(m) for m in train_msgs_raw]

    models = get_models()
    baseline_metrics = {}

    for name, pipeline in models.items():
        print(f"  Training {name}...")
        pipeline.fit(train_msgs, train_labels)
        preds = pipeline.predict(test_msgs)
        metrics = compute_metrics(preds, test_labels, intent_names)
        baseline_metrics[name] = metrics
        print_metrics(metrics, f"{name} Baseline (test set)")

    # Find the best baseline for summary reporting
    best_name = max(baseline_metrics, key=lambda k: baseline_metrics[k]["overall_accuracy"])

    return {
        "metrics":        baseline_metrics,
        "best_baseline":  best_name,
        "train_n":        len(train_msgs),
        "test_n":         len(test_msgs),
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
            "best_baseline_name":     baseline_results["best_baseline"],
            "best_baseline_accuracy": baseline_results["metrics"][baseline_results["best_baseline"]]["overall_accuracy"],
            "agent_accuracy":         clf_metrics["overall_accuracy"],
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
    print(f"  Agent:          {base['agent_accuracy']*100:.1f}%")
    print(f"  Best Baseline ({base['best_baseline_name']}): {base['best_baseline_accuracy']*100:.1f}%")

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
# python3 -m evaluation_harness.automated_metrics
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM judge scoring (faster, classification metrics only)"
    )
    parser.add_argument(
        "--skip-reply", action="store_true",
        help="Skip reply drafting (saves API calls and time)"
    )
    parser.add_argument(
        "--skip-escalate", action="store_true",
        help="Skip LLM escalation checks (saves API calls)"
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
    agent_results = run_agent_eval(
        eval_set, 
        verbose=True, 
        skip_reply=args.skip_reply,
        skip_escalate=args.skip_escalate
    )

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
    print(f"\n  Next: python3 -m evaluation_harness.llm_as_judge (measure human-judge agreement)")