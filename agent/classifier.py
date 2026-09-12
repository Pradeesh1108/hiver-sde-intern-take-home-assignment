# ============================================================
# agent/classifier.py — Intent Classification
# ============================================================
# One public function: classify(message) → intent string
#
# How it works:
#   1. Build classification prompt (from prompts.py)
#   2. Call the LLM
#   3. Validate the response is a known intent name
#   4. Return the intent string
#
# The LLM used here is claude-haiku — fast and cheap.
# We save the more expensive model for reply drafting
# where output quality matters more.
# ============================================================

from agent.prompts import classification_prompt, INTENT_NAMES
from services.llm_factory import generate_completion

# Model choice: haiku is fast + cheap for classification
# max_tokens=20 because the intent name is short (longest is
# "device_hardware_issue" = 21 chars — we give a little headroom)
_MAX_TOKENS = 800

def classify(message: str, retries: int = 2) -> str:
    """
    Classify a customer message into one of the 7 defined intents.

    Args:
        message: cleaned text of the customer's opening tweet
        retries: number of times to retry on API failure

    Returns:
        One of the 7 intent name strings.
        Falls back to "general_inquiry" if something goes wrong.

    Example:
        >>> classify("my battery drains 10% every 5 minutes")
        "battery_drain"
    """
    # Edge case: empty or very short message can't be classified
    if not message or len(message.strip()) < 3:
        return "general_inquiry"

    prompt = classification_prompt(message)

    for attempt in range(retries + 1):
        try:
            # Chain-of-thought response:
            # Line 1: reasoning sentence
            # Line 2: intent name
            # Keep original case for line splitting, lowercase only for matching
            raw = generate_completion(
                messages=[{"role": "user", "content": prompt}],
                task_type="fast",
                max_tokens=_MAX_TOKENS,
                temperature=0.0
            ).strip()

            # Split into lines, take the last non-empty line as the intent
            lines = [line.strip() for line in raw.split('\n') if line.strip()]

            if not lines:
                return "general_inquiry"

            # Last line should be the intent name
            intent_line = lines[-1].lower()

            # Remove punctuation the model might add
            intent_line = intent_line.replace('.', '').replace(',', '').strip()

            if intent_line in INTENT_NAMES:
                return intent_line

            # Fuzzy match — model may have written "Intent: battery_drain"
            for name in INTENT_NAMES:
                if name in intent_line:
                    return name

            # Log full response for debugging
            print(f"  [classifier] Unexpected: '{raw[:100]}' — using general_inquiry")
            return "general_inquiry"
            
        except Exception as e:
            print(f"  [classifier] API error (attempt {attempt + 1}): {e}")
            if attempt < retries:
                import time
                time.sleep(1)   # wait 1 second before retry
            else:
                return "general_inquiry"   # final fallback
# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to verify
# python3 -m agent.classifier
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_messages = [
        "my battery drains 10 percent every 5 minutes since the update",
        "I can't log into my Apple ID, keeps saying wrong password",
        "the screen on my iPhone X cracked after one small drop",
        "iOS 11 is absolute garbage, everything is slow and broken",
        "where is my iPhone X order? it has been 2 weeks",
        "fix this please",   # should be general_inquiry
    ]

    print("Testing classifier...\n")
    for msg in test_messages:
        intent = classify(msg)
        print(f"  [{intent}]")
        print(f"  \"{msg[:70]}\"\n")