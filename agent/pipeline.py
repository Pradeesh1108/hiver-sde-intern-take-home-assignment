# ============================================================
# agent/pipeline.py — The Orchestrator
# ============================================================
# One public function: run(message) → dict
#
# Calls the four agent components in order:
#   1. classify()        → intent
#   2. retrieve()        → similar historical threads
#   3. draft_reply()     → reply grounded in those threads
#   4. should_escalate() → escalation decision + reason
#
# This is the only file that imports from all four components.
# Everything else imports only what it needs.
#
# Run this file directly to test the full pipeline end-to-end:
#   python3 -m agent.pipeline
# ============================================================

import time
import concurrent.futures
from agent.classifier    import classify
from agent.retriever     import retrieve_by_keywords
from agent.reply_drafter import draft_reply
from agent.escalator     import should_escalate
from agent.semantic_retrieve import retrieve_semantic


def run(message: str, verbose: bool = False, skip_reply: bool = False, skip_escalate: bool = False) -> dict:
    """
    Run the full agent pipeline on a single customer message.

    Pipeline:
        message
            → classify()           intent string
            → retrieve()           3 similar historical threads
            → draft_reply()        2-3 sentence reply
            → should_escalate()    AUTO_HANDLE or ESCALATE + reason

    Args:
        message: raw or cleaned customer tweet text
        verbose: if True, print each step's output as it runs

    Returns:
        Dict with:
            message:    the original input message
            intent:     classified intent string
            reply:      drafted reply string
            decision:   "AUTO_HANDLE" or "ESCALATE"
            reason:     why the escalation decision was made
            confidence: escalation confidence score 0.0-1.0
            duration_s: total time taken in seconds

    Example:
        >>> result = run("my battery drains 20% in 10 minutes")
        >>> result["intent"]
        "battery_drain"
        >>> result["decision"]
        "AUTO_HANDLE"
    """
    start_time = time.time()

    if verbose:
        print(f"\n{'='*55}")
        print(f"INPUT: \"{message[:80]}\"")
        print(f"{'='*55}")

    # ── Step 1: Classify ──────────────────────────────────
    intent = classify(message)

    if verbose:
        print(f"[1] Intent:    {intent}")

    # ── Parallel Execution: Escalation & Drafting ──────────
    # The Escalator only needs the message and intent.
    # The Drafter needs the retrieved threads.
    # We can run them both at the same time to save ~1-2 seconds.
    
    def drafting_flow():
        if skip_reply:
            return [], ""
        threads = retrieve_semantic(message, intent, n=3)
        rep = draft_reply(message, intent, threads)
        return threads, rep

    def escalation_flow():
        if skip_escalate:
            return {"decision": "SKIPPED", "reason": "Skipped via --skip-escalate", "confidence": 1.0}
        return should_escalate(message, intent)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_draft = executor.submit(drafting_flow)
        future_escalate = executor.submit(escalation_flow)
        
        similar_threads, reply = future_draft.result()
        escalation = future_escalate.result()

    if verbose:
        print(f"[2] Retrieved: {len(similar_threads)} similar threads")
        print(f"[3] Reply:     \"{reply[:120]}...\"" if len(reply) > 120 else f"[3] Reply: \"{reply}\"")
        print(f"[4] Decision:  {escalation['decision']} — {escalation['reason']}")

    duration = round(time.time() - start_time, 2)

    return {
        "message":    message,
        "intent":     intent,
        "reply":      reply,
        "decision":   escalation["decision"],
        "reason":     escalation["reason"],
        "confidence": escalation.get("confidence", 1.0),
        "duration_s": duration,
    }


def run_batch(messages: list, verbose: bool = False) -> list:
    """
    Run the pipeline on a list of messages.
    Returns a list of result dicts in the same order.

    Used by 04_eval.py to process all 245 eval examples.

    Args:
        messages: list of message strings
        verbose:  if True, print progress every 10 messages

    Returns:
        List of result dicts, one per input message.
    """
    results = []

    for i, message in enumerate(messages):
        result = run(message, verbose=False)
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(messages)}...")

    return results


# ─────────────────────────────────────────────
# QUICK TEST — full pipeline end-to-end
# python3 -m agent.pipeline
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_messages = [
        "my battery drains 20 percent in 10 minutes since the iOS 11 update",
        "I can't log into my Apple ID it says my account is locked",
        "the screen on my iPhone keeps flickering randomly",
        "I'm going to sue Apple if they don't fix this charging issue",
        "iOS 11 is garbage everything is broken and slow",
        "where is my iPhone X order it has been 3 weeks",
    ]

    print("Running full pipeline on test messages...\n")

    for msg in test_messages:
        result = run(msg, verbose=True)
        print(f"  Duration: {result['duration_s']}s")
        print()