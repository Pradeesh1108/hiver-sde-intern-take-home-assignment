# ============================================================
# agent/escalator.py — Escalation Decision
# ============================================================
# One public function: should_escalate(message, intent) → dict
#
# Two-layer approach:
#   Layer 1 — Rule-based hard triggers (instant, no API call)
#     If any hard trigger matches → escalate immediately
#     Hard triggers are things we are 100% sure about:
#     legal threats, safety issues, explicit crisis language
#
#   Layer 2 — LLM judgment for ambiguous cases
#     If no hard trigger → ask the LLM to decide
#     LLM handles nuance: "already tried everything", distress level
#
# Why two layers?
#   Hard rules are fast, free, and 100% reliable for clear cases.
#   LLM handles the grey areas that rules can't enumerate.
#   This is also easier to explain: "here are our hard rules,
#   everything else goes to the LLM with these guidelines."
# ============================================================

import json
import re
import os
from agent.prompts import escalation_prompt
from agent.llm_factory import generate_completion


# ─────────────────────────────────────────────
# LAYER 1: HARD RULE TRIGGERS
# ─────────────────────────────────────────────
# These are checked BEFORE any LLM call.
# If any match → escalate immediately, no API call needed.
# Each trigger has a reason string for the output.

HARD_ESCALATION_TRIGGERS = [
    {
        "name": "legal_threat",
        "patterns": [
            "lawyer", "lawsuit", "sue", "legal action", "court",
            "consumer rights", "trading standards", "attorney"
        ],
        "reason": "Customer mentioned legal action — requires human agent."
    },
    {
        "name": "safety_issue",
        "patterns": [
            "explod", "fire", "burn", "shock", "electr", "smoke",
            "injur", "hospital", "hurt", "danger"
        ],
        "reason": "Customer reported a safety issue — requires immediate human response."
    },
    {
        "name": "crisis_language",
        "patterns": [
            "suicide", "kill myself", "end it", "can't go on",
            "harm myself", "self harm"
        ],
        "reason": "Customer may be in distress — escalate to human with care resources."
    },
]


def _check_hard_triggers(message: str) -> dict | None:
    """
    Check message against hard escalation triggers.

    Returns:
        Dict with decision/reason/confidence if a trigger matches.
        None if no hard trigger matched.
    """
    message_lower = message.lower()

    for trigger in HARD_ESCALATION_TRIGGERS:
        for pattern in trigger["patterns"]:
            if pattern in message_lower:
                return {
                    "decision":   "ESCALATE",
                    "reason":     trigger["reason"],
                    "confidence": 1.0,          # hard rule = 100% confident
                    "trigger":    trigger["name"]  # which rule fired
                }
    return None


# ─────────────────────────────────────────────
# LAYER 2: LLM JUDGMENT
# ─────────────────────────────────────────────

def _llm_escalation_decision(message: str, intent: str) -> dict:
    """
    Ask the LLM to make the escalation decision for ambiguous cases.

    Returns:
        Dict with decision, reason, confidence.
        Falls back to AUTO_HANDLE if parsing fails.
    """
    fallback = {
        "decision":   "AUTO_HANDLE",
        "reason":     "Could not determine escalation need — defaulting to auto-handle.",
        "confidence": 0.5
    }

    prompt = escalation_prompt(message, intent)

    try:
        raw = generate_completion(
            messages=[{"role": "user", "content": prompt}],
            task_type="fast",
            max_tokens=500,
            temperature=0.0
        )

        # Strip markdown code fences if the model added them
        # e.g. ```json {...} ``` → {...}
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        result = json.loads(raw)

        # Validate required fields exist
        if "decision" not in result or "reason" not in result:
            return fallback

        # Validate decision value is one of the two expected strings
        if result["decision"] not in ("AUTO_HANDLE", "ESCALATE"):
            return fallback

        # Safety threshold: if confidence is low, escalate anyway
        # "when in doubt, involve a human"
        confidence = float(result.get("confidence", 1.0))
        if confidence < 0.6 and result["decision"] == "AUTO_HANDLE":
            result["decision"]  = "ESCALATE"
            result["reason"]    = f"Low confidence ({confidence:.2f}) in auto-handle — escalating for safety."
            result["confidence"] = confidence

        return result

    except json.JSONDecodeError:
        print(f"  [escalator] Failed to parse JSON response: {raw[:80]}")
        return fallback

    except Exception as e:
        print(f"  [escalator] Error: {e}")
        return fallback


# ─────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────

def should_escalate(message: str, intent: str) -> dict:
    """
    Decide whether a customer message should be escalated to a human.

    Two-layer approach:
        1. Check hard rule triggers (instant, no API)
        2. If no hard trigger, ask LLM for judgment

    Args:
        message: cleaned customer tweet text
        intent:  classified intent string

    Returns:
        Dict with:
            decision:   "AUTO_HANDLE" or "ESCALATE"
            reason:     one sentence explaining why
            confidence: 0.0 to 1.0
            trigger:    which hard rule fired (only present for layer 1)

    Example:
        >>> should_escalate("I'm going to sue Apple for this", "order_purchase")
        {
            "decision": "ESCALATE",
            "reason": "Customer mentioned legal action — requires human agent.",
            "confidence": 1.0,
            "trigger": "legal_threat"
        }
    """
    # Layer 1: hard rules first — fast and free
    hard_result = _check_hard_triggers(message)
    if hard_result:
        return hard_result

    # Layer 2: LLM judgment for everything else
    return _llm_escalation_decision(message, intent)


# ─────────────────────────────────────────────
# QUICK TEST
# python3 -m agent.escalator
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("I'm going to sue Apple if this isn't fixed today", "order_purchase"),
        ("my charger literally caught fire last night", "battery_drain"),
        ("my battery drains fast since the update", "ios_update_issue"),
        ("I've tried everything, reset, reinstall, nothing works, this is ridiculous", "app_issue"),
        ("how do I update my iOS?", "general_inquiry"),
    ]

    print("Testing escalator...\n")
    for message, intent in test_cases:
        result = should_escalate(message, intent)
        print(f"  Message:    \"{message[:65]}\"")
        print(f"  Intent:     {intent}")
        print(f"  Decision:   {result['decision']}")
        print(f"  Reason:     {result['reason']}")
        print(f"  Confidence: {result['confidence']}")
        print()