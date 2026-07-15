from langchain_openai import ChatOpenAI
import json
import re
import httpx

# -------------------------
# LLM CONFIG (WORKING WITH YOUR ENV)
# -------------------------
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-4P6CDApaILnr6w_9NVM8Cw",
    http_client=client,
    temperature=0.1
)

# -------------------------
# BUILD CLEAN CONTEXT FROM RAG
# -------------------------
def build_context_from_rag(rag_output: dict) -> str:
    """
    Converts RAG JSON into structured text for LLM
    """

    context_parts = []

    # Requirements
    for r in rag_output.get("requirements", []):
        text = r.get("text", "")
        if text:
            context_parts.append(f"[REQUIREMENT]\n{text}")

    # Emails
    for e in rag_output.get("emails", []):
        text = e.get("text", "")
        if text:
            context_parts.append(f"[EMAIL]\n{text}")

    # Transcripts
    for t in rag_output.get("transcripts", []):
        text = t.get("text", "")
        if text:
            context_parts.append(f"[TRANSCRIPT]\n{text}")

    return "\n\n".join(context_parts)


# -------------------------
# MAIN FUNCTION
# -------------------------
def generate_answer(query: str, rag_output: dict, history: list = None, mode: str = "clarify") -> str:

    # -------------------------
    # STEP 1: CONTEXT + HISTORY
    # -------------------------
    context_text = build_context_from_rag(rag_output)

    history_text = ""
    if history and isinstance(history, list):
        history_text = "\n".join([str(h) for h in history])

    full_input = f"""
Conversation History:
{history_text}

Latest Input:
{query}
"""

    # -------------------------
    # STEP 2: PROMPT
    # -------------------------
    if mode == "clarify":

        prompt = f"""
You are a senior business analyst AI specializing in requirement clarification.

Use the conversation and context below to:
- detect ambiguity
- ask clarifying questions
- refine requirements

----------------------------------------
CONVERSATION
----------------------------------------
{full_input}

----------------------------------------
RETRIEVED CONTEXT (use only if relevant)
----------------------------------------
{context_text}

----------------------------------------
OUTPUT FORMAT (STRICT JSON)
----------------------------------------
{{
  "inference": "brief summary",
  "cqs": ["question1", "question2", "..."],
  "ambiguity_score": 0,
  "reasoning": "why ambiguous",
  "confidence_score": 0.0
}}

----------------------------------------
RULES
----------------------------------------
- ambiguity_score must be integer (0-10)
- confidence_score must be float (0-1)
- minimum 3 questions if ambiguity_score > 5
- DO NOT repeat previous questions
- DO NOT hallucinate
- OUTPUT ONLY JSON

Return ONLY JSON.
"""

    else:
        prompt = f"""
You are a senior business analyst AI.

Generate a FINAL Software Requirement Specification (SRS).

----------------------------------------
CONVERSATION
----------------------------------------
{full_input}

----------------------------------------
CONTEXT
----------------------------------------
{context_text}

----------------------------------------
OUTPUT FORMAT (STRICT JSON)
----------------------------------------
{{
  "final_srs": "structured SRS document"
}}

----------------------------------------
RULES
----------------------------------------
- use only provided information
- no assumptions
- structured format
- OUTPUT ONLY JSON

Return ONLY JSON.
"""

    # -------------------------
    # STEP 3: CALL LLM
    # -------------------------
    try:
        response = llm.invoke(prompt)
        raw_content = str(response.content).strip()

    except Exception as e:
        return json.dumps({
            "inference": "LLM invocation failed",
            "cqs": ["Retry request"],
            "ambiguity_score": 10,
            "reasoning": str(e),
            "confidence_score": 0.5
        })

    # -------------------------
    # STEP 4: SAFE JSON EXTRACTION
    # -------------------------
    try:
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            raw_content = match.group(0)

        parsed = json.loads(raw_content)

        # -------------------------
        # STEP 5: POST PROCESS
        # -------------------------
        if mode == "clarify":

            # Ensure fields exist
            parsed.setdefault("inference", "")
            parsed.setdefault("cqs", [])
            parsed.setdefault("ambiguity_score", 10)
            parsed.setdefault("reasoning", "")
            parsed.setdefault("confidence_score", 0.5)

            # Normalize ambiguity
            try:
                score = int(parsed["ambiguity_score"])
            except:
                score = 10

            score = max(0, min(10, score))
            parsed["ambiguity_score"] = score

            # Deterministic confidence
            if score >= 7:
                parsed["confidence_score"] = 0.6
            elif score >= 4:
                parsed["confidence_score"] = 0.75
            else:
                parsed["confidence_score"] = 0.9

            # Ensure minimum questions
            if score > 5 and len(parsed["cqs"]) < 3:
                parsed["cqs"].append("Can you provide more detailed functional requirements?")

        else:
            if "final_srs" not in parsed:
                raise ValueError("Missing final_srs")

        return json.dumps(parsed)

    except Exception as e:
        return json.dumps({
            "inference": "Parsing failed",
            "cqs": [
                "What is the main goal?",
                "Who are the users?",
                "What inputs/outputs exist?"
            ],
            "ambiguity_score": 9,
            "reasoning": str(e),
            "confidence_score": 0.6
        })


# -------------------------
# LOCAL TEST
# -------------------------
if __name__ == "__main__":
    test_rag = {
        "requirements": [{"text": "System should allow login"}],
        "emails": [{"text": "Should OTP be included?"}],
        "transcripts": [{"text": "Stakeholder wants email login"}]
    }

    print(generate_answer("We need login system", test_rag))