import pandas as pd
import re


# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize_text(text):
    """
    Lowercase + clean spacing + remove unwanted characters
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text).lower()

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # keep only useful characters
    text = re.sub(r"[^a-z0-9.,!?@ ]", "", text)

    return text.strip()


# -------------------------------
# ANONYMIZATION
# -------------------------------
def anonymize_text(text):
    """
    Remove sensitive patterns like emails, phone numbers, names
    """
    if not text:
        return ""

    text = str(text)

    # Emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)

    # Phone numbers (10 digits)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)

    # Names (simple heuristic: First Last)
    text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME]", text)

    return text


# -------------------------------
# FULL PREPROCESSING PIPELINE
# -------------------------------
def preprocess_text(text):
    """
    Apply normalization + anonymization
    """
    text = normalize_text(text)
    text = anonymize_text(text)
    return text


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    """
    Loads CSV files and validates structure
    """

    try:
        req = pd.read_csv("srs_consistent.csv")
        emails = pd.read_csv("emails_consistent.csv")
        transcripts = pd.read_csv("transcripts_consistent.csv")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load CSV files: {e}")

    # -------------------------------
    # VALIDATION
    # -------------------------------
    required_req_cols = {"id", "srs"}
    required_email_cols = {"id", "email_subject", "email_body"}
    required_trans_cols = {"id", "transcript"}

    if not required_req_cols.issubset(req.columns):
        raise ValueError(f"❌ SRS file missing columns: {required_req_cols}")

    if not required_email_cols.issubset(emails.columns):
        raise ValueError(f"❌ Emails file missing columns: {required_email_cols}")

    if not required_trans_cols.issubset(transcripts.columns):
        raise ValueError(f"❌ Transcripts file missing columns: {required_trans_cols}")

    # -------------------------------
    # CLEAN EMPTY VALUES
    # -------------------------------
    req = req.fillna("")
    emails = emails.fillna("")
    transcripts = transcripts.fillna("")

    # -------------------------------
    # ENSURE TYPES
    # -------------------------------
    req["id"] = req["id"].astype(int)
    emails["id"] = emails["id"].astype(int)
    transcripts["id"] = transcripts["id"].astype(int)

    print("✅ Data loaded successfully")
    print(f"📄 SRS records: {len(req)}")
    print(f"📧 Email records: {len(emails)}")
    print(f"🗣️ Transcript records: {len(transcripts)}")

    return {
        "requirements": req,
        "emails": emails,
        "transcripts": transcripts
    }


# -------------------------------
# LOCAL TEST
# -------------------------------
if __name__ == "__main__":
    data = load_data()

    print("\n--- SAMPLE OUTPUT ---")
    print(data["requirements"].head(2))
    print(data["emails"].head(2))
    print(data["transcripts"].head(2))