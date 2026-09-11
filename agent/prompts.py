# ============================================================
# agent/prompts.py — All LLM prompts in one place
# ============================================================
# Every prompt used by the agent lives here.
# No prompt strings should appear in any other file.
#
# Each prompt is a function that takes arguments and returns
# a fully-formed string ready to send to the LLM.
# Functions are easier to test than raw f-strings scattered
# across multiple files.
# ============================================================

import json


# ─────────────────────────────────────────────
# LOAD SHARED DATA (loaded once at import time)
# ─────────────────────────────────────────────
# We load the intent taxonomy and few-shot examples once
# when this module is imported, not on every function call.
# This avoids re-reading the same files thousands of times
# during the eval run.

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Paths relative to project root (where you run python from)
_taxonomy    = _load_json("datasets/intent_taxonomy.json")
_few_shot    = _load_json("datasets/few_shot_examples.json")

INTENTS      = _taxonomy["intents"]           # list of intent dicts
INTENT_NAMES = _taxonomy["intent_names"]      # list of intent name strings
FEW_SHOT     = _few_shot["few_shot_examples"] # dict: intent_name → [examples]


# ─────────────────────────────────────────────
# PROMPT 1: CLASSIFICATION
# ─────────────────────────────────────────────
# Used by: classifier.py
#
# Design decisions:
#   - Show intent name + description + real examples so the model
#     understands both the definition AND what it looks like in practice
#   - "Reply with ONLY the intent name" forces clean output we can
#     store directly without parsing
#   - Few-shot examples are from real Apple Support data —
#     more representative than invented examples

def classification_prompt(customer_message: str) -> str:
    """
    Build the prompt for classifying one customer message into an intent.

    Args:
        customer_message: cleaned text of the customer's opening tweet

    Returns:
        Fully-formed prompt string.
        Expected LLM response: exactly one intent name, nothing else.
    """
    intent_lines = []
    for intent in INTENTS:
        name     = intent["name"]
        desc     = intent["description"]
        examples = FEW_SHOT.get(name, [])
        ex_str   = " | ".join(f'"{e[:60]}"' for e in examples[:3])

        intent_lines.append(
            f"- {name}\n"
            f"  Definition: {desc}\n"
            f"  Examples: {ex_str}"
        )

    intent_section = "\n\n".join(intent_lines)

    return f"""You are classifying customer support messages...

        {intent_section}

        BOUNDARY RULES — apply these before deciding:
        1. Phone FROZEN, BLACK SCREEN, WON'T TURN ON → phone_freezing
        (even if an update caused it — the symptom is the freeze)
        2. Cannot log into APPLE ID or ICLOUD → account_access
        (not app_issue, even if the problem is inside an app)
        3. CHARGED wrongly, BILLING issue, ORDER problem → order_purchase
        (not account_access, even if the account is involved)
        4. Update caused a SPECIFIC symptom → use the specific intent
        "update broke my wifi" → wifi_bluetooth (not ios_update_general)
        "update killed my battery" → battery_drain (not ios_update_general)
        "phone frozen since update" → phone_freezing (not ios_update_general)
        ios_update_general only when no more specific intent fits.

        Customer message:
        \"\"\"{customer_message}\"\"\"

        Step 1 — In one sentence, identify the PRIMARY problem the customer has.
        Step 2 — Write the intent name on the next line.

        Your response must be exactly two lines:
        Line 1: one sentence describing the primary problem
        Line 2: the intent name (nothing else)"""


# ─────────────────────────────────────────────
# PROMPT 2: REPLY DRAFTING
# ─────────────────────────────────────────────
# Used by: reply_drafter.py
#
# Design decisions:
#   - Pass 3 similar historical threads as grounding so the reply
#     sounds like Apple, not a generic chatbot
#   - State Apple's tone explicitly: empathetic, concise, action-oriented
#   - Cap reply at 2-3 sentences — Twitter replies are short
#   - "Do not copy verbatim" prevents lazy reproduction of past replies

def reply_drafting_prompt(
    customer_message: str,
    intent: str,
    similar_threads: list
) -> str:
    """
    Build the prompt for drafting a reply to a customer message.

    Args:
        customer_message: cleaned customer tweet text
        intent:           classified intent string
        similar_threads:  list of dicts from retriever.py, each with
                          'customer_msg' and 'brand_replies' keys

    Returns:
        Fully-formed prompt string.
        Expected LLM response: 2-3 sentence reply in Apple's tone.
    """
    example_lines = []
    for i, thread in enumerate(similar_threads[:3], 1):
        past_customer = thread.get("customer_msg", "")[:120]
        past_replies  = thread.get("brand_replies", [])
        past_reply    = past_replies[0] if past_replies else ""

        example_lines.append(
            f"Example {i}:\n"
            f"  Customer: \"{past_customer}\"\n"
            f"  Apple replied: \"{past_reply[:150]}\""
        )

    examples_section = (
        "\n\n".join(example_lines)
        if example_lines
        else "No similar historical examples found."
    )

    return f"""You are an Apple Support agent responding to a customer tweet.

The customer's issue has been classified as: {intent}

Here are examples of how Apple Support has handled similar {intent} issues in the past:

{examples_section}

Now write a reply to this customer:
\"\"\"{customer_message}\"\"\"

Guidelines:
- Match Apple Support's tone: empathetic, professional, concise
- Keep it to 2-3 sentences maximum (Twitter reply length)
- Acknowledge the specific problem the customer mentioned
- Ask ONE clarifying question OR suggest ONE concrete next step
- Do not copy the example replies verbatim — adapt them to this situation
- Do not use the customer's name (it is anonymised in this dataset)
- If troubleshooting is needed, invite them to DM for privacy

Write only the reply. No preamble. No sign-off."""


# ─────────────────────────────────────────────
# PROMPT 3: ESCALATION DECISION
# ─────────────────────────────────────────────
# Used by: escalator.py
#
# Design decisions:
#   - Define clear escalation triggers explicitly so the decision
#     is rule-grounded, not vague LLM intuition
#   - Ask for JSON {decision, reason, confidence} so the eval
#     script can parse it programmatically
#   - Confidence score lets us set a safety threshold:
#     confidence < 0.6 → escalate (when unsure, involve a human)
#   - "reason" field is mandatory — the assignment explicitly
#     requires a stated reason for every escalation decision

def escalation_prompt(customer_message: str, intent: str) -> str:
    """
    Build the prompt for deciding whether to escalate to a human agent.

    Args:
        customer_message: cleaned customer tweet text
        intent:           classified intent string

    Returns:
        Fully-formed prompt string.
        Expected LLM response: JSON with decision, reason, confidence.
    """
    return f"""You are a triage system for Apple Support on Twitter.

Decide whether this customer message should be:
  - AUTO_HANDLE: the support agent can respond automatically
  - ESCALATE: a human agent must handle this personally

Escalate if ANY of these are true:
1. Customer mentions legal action, lawsuit, or consumer rights violation
2. Customer mentions a safety issue (fire, explosion, injury, electric shock)
3. Customer is severely distressed or uses crisis language
4. Financial dispute over a significant amount (wrong charge, refund refused multiple times)
5. Customer has already tried all standard troubleshooting steps and the issue persists
6. The message is ambiguous enough that a wrong automated reply could cause real harm
7. The message contains threats or abusive language requiring human judgement

Auto-handle if:
- Common, well-understood problem with standard troubleshooting steps available
- Customer asking a general question or product question
- Customer has not yet tried basic troubleshooting
- Routine issue the agent can resolve by directing to DM

Customer message: \"\"\"{customer_message}\"\"\"
Classified intent: {intent}

Respond with ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "decision": "AUTO_HANDLE" or "ESCALATE",
  "reason": "one sentence explaining the decision",
  "confidence": 0.0 to 1.0
}}"""


# ─────────────────────────────────────────────
# PROMPT 4: LLM-AS-JUDGE (reply quality)
# ─────────────────────────────────────────────
# Used by: eval/judge.py
#
# Design decisions:
#   - Score on 4 dimensions separately, not one overall score —
#     gives richer diagnostic signal for failure analysis
#   - Dimensions match what the assignment evaluates:
#     relevance, tone, actionability, conciseness
#   - Scale 1-5 not 1-10 — less ambiguity between adjacent scores
#   - JSON output for programmatic parsing in 04_eval.py

def judge_prompt(
    customer_message: str,
    intent: str,
    agent_reply: str
) -> str:
    """
    Build the prompt for LLM-as-judge to evaluate a drafted reply.

    Args:
        customer_message: the original customer tweet
        intent:           the classified intent
        agent_reply:      the reply drafted by reply_drafter.py

    Returns:
        Fully-formed prompt string.
        Expected LLM response: JSON with four 1-5 scores + reasoning.
    """
    return f"""You are evaluating the quality of an Apple Support reply on Twitter.

Customer message: \"\"\"{customer_message}\"\"\"
Classified intent: {intent}
Agent reply: \"\"\"{agent_reply}\"\"\"

Score the reply on these 4 dimensions (1 = very poor, 5 = excellent):

1. Relevance    — Does it directly address the customer's specific problem?
2. Tone         — Does it sound like a professional, empathetic Apple Support agent?
3. Actionability — Does it give the customer a clear next step or ask a useful question?
4. Conciseness  — Is it appropriately brief for a Twitter reply (2-3 sentences)?

Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "relevance": 1-5,
  "tone": 1-5,
  "actionability": 1-5,
  "conciseness": 1-5,
  "overall": 1-5,
  "reasoning": "one sentence explaining the overall score"
}}"""