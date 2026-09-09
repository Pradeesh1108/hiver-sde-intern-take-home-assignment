# ============================================================
# agent/reply_drafter.py — Reply Drafting
# ============================================================
# One public function: draft_reply(message, intent, threads) → str
#
# How it works:
#   1. Takes the customer message, classified intent, and
#      similar historical threads from retriever.py
#   2. Builds the reply drafting prompt (from prompts.py)
#   3. Calls the LLM
#   4. Returns the drafted reply string
#
# Model choice: claude-sonnet for reply drafting (not haiku)
# because reply quality matters — this is customer-facing text.
# Classification uses haiku (speed/cost), reply uses sonnet (quality).
# ============================================================

import os
from dotenv import load_dotenv
from groq import Groq
from agent.prompts import reply_drafting_prompt

load_dotenv()
_client     = Groq()
_MODEL      = "openai/gpt-oss-120b"   # better quality for customer-facing text
_MAX_TOKENS = 256                    # 2-3 sentences fits comfortably


def draft_reply(
    message: str,
    intent: str,
    similar_threads: list,
    retries: int = 2
) -> str:
    """
    Draft a reply to a customer message grounded in historical examples.

    Args:
        message:         cleaned customer tweet text
        intent:          classified intent string
        similar_threads: list of historical threads from retriever.py
        retries:         number of API retry attempts on failure

    Returns:
        A 2-3 sentence reply string in Apple Support's tone.
        Returns a safe fallback message if the API fails.

    Example:
        >>> threads = retrieve("battery_drain", n=3)
        >>> draft_reply("my battery dies in 2 hours", "battery_drain", threads)
        "We're sorry to hear your battery isn't lasting as long as expected.
         Could you let us know which iPhone model you have and what iOS
         version you're running? DM us and we'll take a closer look together."
    """
    # Safety fallback — if something goes wrong, return a generic
    # but safe response rather than crashing the pipeline
    fallback = (
        "We're sorry to hear you're having trouble. "
        "Please DM us with more details about your issue "
        "and we'll be happy to help you get this sorted out."
    )

    if not message or not message.strip():
        return fallback

    prompt = reply_drafting_prompt(message, intent, similar_threads)

    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            reply = response.choices[0].message.content.strip()

            # Basic sanity check: reply should not be empty
            # and should not be longer than ~300 chars (Twitter limit ~280)
            if not reply:
                return fallback
                
            if len(reply) < 50:
                print(f"  [reply_drafter] Reply too short ({len(reply)} chars): {reply}")
                return fallback

            # Trim if somehow over Twitter length
            # (shouldn't happen with max_tokens=256 but just in case)
            if len(reply) > 280:
                reply = reply[:277] + "..."

            return reply

        except Exception as e:
            print(f"  [reply_drafter] API error (attempt {attempt + 1}): {e}")
            if attempt < retries:
                import time
                time.sleep(1)
            else:
                return fallback

        except Exception as e:
            print(f"  [reply_drafter] Unexpected error: {e}")
            return fallback


# ─────────────────────────────────────────────
# QUICK TEST
# python3 -m agent.reply_drafter
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from agent.retriever import retrieve_by_keywords

    test_cases = [
        ("my battery drains 20 percent in 20 minutes wtf", "battery_drain"),
        ("iOS 11 broke everything apps crash constantly", "ios_update_issue"),
        ("can't log into my Apple ID says account is locked", "account_access"),
    ]

    print("Testing reply drafter...\n")
    for message, intent in test_cases:
        threads = retrieve_by_keywords(message, intent, n=3)
        reply = draft_reply(message, intent, threads)

        print(f"  Intent:  {intent}")
        print(f"  Message: \"{message}\"")
        print(f"  Reply:   \"{reply}\"")
        print(f"  Length:  {len(reply)} chars")
        print()