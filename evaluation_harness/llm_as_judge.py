# ============================================================
# evaluation_harness/llm_as_judge.py — LLM-as-Judge for Reply Quality
# ============================================================
# One public function: judge_reply(message, intent, reply) → dict
#
# The judge reads a customer message and the agent's drafted reply,
# then scores the reply on 4 dimensions (1-5 each):
#   relevance, tone, actionability, conciseness
#
# Why LLM-as-judge?
#   Human evaluation of 245 replies would take hours and is
#   subjective. An LLM judge is faster, consistent, and can be
#   calibrated against human scores to measure its own reliability.
#   The assignment explicitly asks for "evidence of how well your
#   judge agrees with a human" — we handle that with an agreement
#   measurement on a small sample.
#
# Important limitation:
#   LLM-as-judge has known biases:
#   - Prefers longer replies (verbosity bias)
#   - Prefers its own outputs (self-preference bias)
#   - May be inconsistent on borderline cases
#   These are documented in the report's "what is misleading" section.
# ============================================================

import json
import re
import os
from agent.prompts import judge_prompt
from agent.llm_factory import generate_completion


def judge_reply(
    customer_message: str,
    intent: str,
    agent_reply: str,
    retries: int = 2
) -> dict:
    """
    Score an agent reply on 4 quality dimensions using an LLM judge.

    Args:
        customer_message: the original customer tweet
        intent:           the classified intent
        agent_reply:      the reply drafted by reply_drafter.py
        retries:          API retry attempts

    Returns:
        Dict with scores (1-5) for each dimension:
            relevance:     does it address the specific problem?
            tone:          does it sound like Apple Support?
            actionability: does it give a clear next step?
            conciseness:   is it Twitter-appropriate length?
            overall:       judge's holistic score
            reasoning:     one sentence explanation
            error:         only present if scoring failed

    Example:
        >>> judge_reply("battery dies in 2 hours", "battery_drain",
        ...             "We're sorry to hear that. What iOS version?")
        {"relevance": 4, "tone": 5, "actionability": 4,
         "conciseness": 5, "overall": 4, "reasoning": "..."}
    """
    # Fallback scores if the API fails
    # We use None to signal "could not score" rather than inventing numbers
    fallback = {
        "relevance":     None,
        "tone":          None,
        "actionability": None,
        "conciseness":   None,
        "overall":       None,
        "reasoning":     "Scoring failed — API error",
        "error":         True,
    }

    prompt = judge_prompt(customer_message, intent, agent_reply)

    for attempt in range(retries + 1):
        try:
            raw = generate_completion(
                messages=[{"role": "user", "content": prompt}],
                task_type="fast",
                max_tokens=500,
                temperature=0.0
            )

            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip()

            scores = json.loads(raw)

            # Validate all required keys are present
            required = ["relevance", "tone", "actionability", "conciseness", "overall"]
            if not all(k in scores for k in required):
                raise ValueError(f"Missing keys in judge response: {raw[:80]}")

            # Validate all scores are in range 1-5
            for key in required:
                val = scores[key]
                if not isinstance(val, (int, float)) or not (1 <= val <= 5):
                    raise ValueError(f"Score out of range for {key}: {val}")

            return scores

        except json.JSONDecodeError:
            print(f"  [judge] JSON parse error (attempt {attempt+1}): {raw[:60]}")
            if attempt < retries:
                import time
                time.sleep(1)
            else:
                return fallback

        except Exception as e:
            print(f"  [judge] Error (attempt {attempt+1}): {e}")
            if attempt < retries:
                import time
                time.sleep(1)
            else:
                return fallback


def judge_batch(examples: list, verbose: bool = False) -> list:
    """
    Score a list of examples through the judge.

    Args:
        examples: list of dicts, each with:
                  'message', 'intent', 'reply' keys
        verbose:  print progress every 10 examples

    Returns:
        List of score dicts in the same order as input.
    """
    results = []
    for i, ex in enumerate(examples):
        scores = judge_reply(
            ex["message"],
            ex["intent"],
            ex["reply"]
        )
        results.append(scores)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Judged {i+1}/{len(examples)}...")

    return results


def measure_human_agreement(sample_size: int = 30) -> dict:
    """
    Measure how well the LLM judge agrees with human scores.

    This is what the assignment means by "evidence of how well
    your judge agrees with a human."

    Process:
        1. Take sample_size replies from the agent's eval output
        2. Human (you) scores them on the same 1-5 rubric
        3. Compare human scores vs judge scores
        4. Compute agreement rate and mean absolute error

    Args:
        sample_size: how many replies to have the human score

    Returns:
        Dict with agreement statistics.

    NOTE: This function prints instructions for human scoring.
          It does not automate the human scoring step —
          that requires you to read and score the replies yourself.
    """
    import os
    import csv

    # Check if agent eval results exist
    if not os.path.exists("outputs/eval_results.json"):
        print("  Run automated_metrics.py first to generate eval_results.json")
        return {}

    with open("outputs/eval_results.json") as f:
        eval_results = json.load(f)

    agent_results = eval_results.get("agent_results", [])

    if len(agent_results) < sample_size:
        sample_size = len(agent_results)

    import random
    random.seed(42)
    sample = random.sample(agent_results, sample_size)

    # Save sample to CSV for human scoring
    human_scoring_path = "outputs/human_scoring_sample.csv"
    with open(human_scoring_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "message", "intent", "reply",
            "judge_relevance", "judge_tone", "judge_actionability",
            "judge_conciseness", "judge_overall",
            "human_relevance", "human_tone", "human_actionability",
            "human_conciseness", "human_overall"
        ])
        writer.writeheader()
        for i, result in enumerate(sample):
            scores = result.get("judge_scores", {})
            writer.writerow({
                "id":                  i + 1,
                "message":             result.get("message", ""),
                "intent":              result.get("intent", ""),
                "reply":               result.get("reply", ""),
                "judge_relevance":     scores.get("relevance", ""),
                "judge_tone":          scores.get("tone", ""),
                "judge_actionability": scores.get("actionability", ""),
                "judge_conciseness":   scores.get("conciseness", ""),
                "judge_overall":       scores.get("overall", ""),
                # Human fills these in
                "human_relevance":     "",
                "human_tone":          "",
                "human_actionability": "",
                "human_conciseness":   "",
                "human_overall":       "",
            })

    print(f"\n  Saved {sample_size} replies for human scoring → {human_scoring_path}")
    print()
    print("  HOW TO SCORE:")
    print("  1. Open human_scoring_sample.csv")
    print("  2. Read each 'message' and 'reply'")
    print("  3. Score each reply 1-5 on the human_* columns")
    print("  4. Use the same rubric as the judge:")
    print("     relevance:     does it address the customer's specific problem?")
    print("     tone:          does it sound like Apple Support?")
    print("     actionability: does it give a clear next step?")
    print("     conciseness:   is it Twitter-appropriate length?")
    print("     overall:       your holistic 1-5 score")
    print()
    print("  After scoring, run compute_agreement() to see judge vs human stats.")

    return {"sample_path": human_scoring_path, "sample_size": sample_size}


def compute_agreement(human_scoring_path: str = "outputs/human_scoring_sample.csv") -> dict:
    """
    After human scoring is done, compute agreement statistics.

    Reads the filled-in human_scoring_sample.csv and computes:
        - Mean Absolute Error per dimension
        - Agreement rate (within 1 point)
        - Pearson correlation per dimension

    Args:
        human_scoring_path: path to the filled-in CSV

    Returns:
        Dict with agreement statistics per dimension.
    """
    import csv
    import os

    if not os.path.exists(human_scoring_path):
        print(f"  File not found: {human_scoring_path}")
        print("  Run measure_human_agreement() first, then fill in human scores.")
        return {}

    dimensions = ["relevance", "tone", "actionability", "conciseness", "overall"]
    human_scores = {d: [] for d in dimensions}
    judge_scores = {d: [] for d in dimensions}

    with open(human_scoring_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows where human hasn't scored yet
            if not row.get("human_overall", "").strip():
                continue
            try:
                for d in dimensions:
                    human_scores[d].append(float(row[f"human_{d}"]))
                    judge_scores[d].append(float(row[f"judge_{d}"]))
            except (ValueError, KeyError):
                continue

    if not human_scores["overall"]:
        print("  No completed human scores found in the CSV.")
        return {}

    results = {}
    print(f"\n  Judge vs Human Agreement ({len(human_scores['overall'])} examples)")
    print(f"  {'Dimension':<15} {'MAE':>6} {'Within-1':>10} {'Correlation':>12}")
    print(f"  {'─'*50}")

    for d in dimensions:
        h = human_scores[d]
        j = judge_scores[d]
        n = len(h)

        mae      = sum(abs(hi - ji) for hi, ji in zip(h, j)) / n
        within_1 = sum(abs(hi - ji) <= 1 for hi, ji in zip(h, j)) / n

        # Pearson correlation
        h_mean = sum(h) / n
        j_mean = sum(j) / n
        num    = sum((hi - h_mean) * (ji - j_mean) for hi, ji in zip(h, j))
        h_std  = (sum((hi - h_mean)**2 for hi in h) / n) ** 0.5
        j_std  = (sum((ji - j_mean)**2 for ji in j) / n) ** 0.5
        corr   = (num / (n * h_std * j_std)) if (h_std > 0 and j_std > 0) else 0.0

        results[d] = {"mae": mae, "within_1": within_1, "correlation": corr}
        print(f"  {d:<15} {mae:>6.2f} {within_1*100:>9.1f}% {corr:>12.3f}")

    return results


# ─────────────────────────────────────────────
# QUICK TEST
# python3 -m evaluation_harness.llm_as_judge
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {
            "message": "my battery drains 20 percent in 10 minutes since the iOS 11 update",
            "intent":  "ios_update_issue",
            "reply":   "We're sorry to hear iOS 11 is affecting your battery life. "
                       "Could you DM us your iPhone model and the exact iOS version? "
                       "We'd love to help troubleshoot this together."
        },
        {
            "message": "fix this",
            "intent":  "general_inquiry",
            "reply":   "We're here to help! Could you tell us a bit more about "
                       "what's happening so we can point you in the right direction?"
        },
    ]

    print("Testing judge...\n")
    for ex in test_cases:
        scores = judge_reply(ex["message"], ex["intent"], ex["reply"])
        print(f"  Message: \"{ex['message'][:60]}\"")
        print(f"  Reply:   \"{ex['reply'][:60]}\"")
        print(f"  Scores:  relevance={scores.get('relevance')} "
              f"tone={scores.get('tone')} "
              f"actionability={scores.get('actionability')} "
              f"conciseness={scores.get('conciseness')} "
              f"overall={scores.get('overall')}")
        print(f"  Reason:  {scores.get('reasoning', '')}")
        print()