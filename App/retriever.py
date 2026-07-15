from vector_store import db


def retrieve(query: str, k: int = 6):
    """
    Retrieves relevant documents from vector DB
    and returns structured JSON output.

    Output format:
    {
        "query": "...",
        "requirements": [{"requirement_id": ..., "text": "..."}],
        "emails": [{"requirement_id": ..., "text": "..."}],
        "transcripts": [{"requirement_id": ..., "text": "..."}]
    }
    """

    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if db is None:
        return {
            "query": query,
            "requirements": [],
            "emails": [],
            "transcripts": []
        }

    try:
        docs = db.similarity_search(query, k=k)
    except Exception:
        # fallback if search fails
        return {
            "query": query,
            "requirements": [],
            "emails": [],
            "transcripts": []
        }

    # -------------------------
    # STRUCTURED RESULT
    # -------------------------
    result = {
        "query": query,
        "requirements": [],
        "emails": [],
        "transcripts": []
    }

    # -------------------------
    # PROCESS DOCUMENTS
    # -------------------------
    for d in docs:
        try:
            src = d.metadata.get("source", "")
            rid = d.metadata.get("requirement_id", None)
            text = d.page_content or ""

            if not text:
                continue

            entry = {
                "requirement_id": rid,
                "text": text
            }

            if src == "requirement":
                result["requirements"].append(entry)

            elif src == "email":
                result["emails"].append(entry)

            elif src == "transcript":
                result["transcripts"].append(entry)

        except Exception:
            continue  # skip bad document safely

    return result


# -------------------------
# LOCAL TEST
# -------------------------
if __name__ == "__main__":
    print(retrieve("login system"))