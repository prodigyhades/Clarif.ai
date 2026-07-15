from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import httpx

from data_loader import preprocess_text

# -------------------------
# EMBEDDING MODEL CONFIG
# -------------------------
client = httpx.Client(verify=False)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-4P6CDApaILnr6w_9NVM8Cw",
    http_client=client
)

# Global DB reference
db = None


# -------------------------
# BUILD DOCUMENTS
# -------------------------
def build_documents(data):
    docs = []

    req_df = data["requirements"]
    email_df = data["emails"]
    transcript_df = data["transcripts"]

    for _, row in req_df.iterrows():
        req_id = row.get("id")

        # -------------------------
        # REQUIREMENT
        # -------------------------
        req_text = preprocess_text(row.get("srs", ""))

        if req_text:
            docs.append(Document(
                page_content=req_text,
                metadata={
                    "requirement_id": req_id,
                    "source": "requirement"
                }
            ))

        # -------------------------
        # EMAILS (linked by id)
        # -------------------------
        email_rows = email_df[email_df["id"] == req_id]

        for _, e in email_rows.iterrows():
            subject = preprocess_text(e.get("email_subject", ""))
            body = preprocess_text(e.get("email_body", ""))

            email_text = f"Subject: {subject}\nBody: {body}".strip()

            if email_text:
                docs.append(Document(
                    page_content=email_text,
                    metadata={
                        "requirement_id": req_id,
                        "source": "email"
                    }
                ))

        # -------------------------
        # TRANSCRIPTS
        # -------------------------
        transcript_rows = transcript_df[transcript_df["id"] == req_id]

        for _, t in transcript_rows.iterrows():
            t_text = preprocess_text(t.get("transcript", ""))

            if t_text:
                docs.append(Document(
                    page_content=t_text,
                    metadata={
                        "requirement_id": req_id,
                        "source": "transcript"
                    }
                ))

    return docs


# -------------------------
# INDEX DATA INTO VECTOR DB
# -------------------------
def index_data(data):
    global db

    print("📦 Building documents...")

    docs = build_documents(data)

    print(f"📄 Total raw documents: {len(docs)}")

    # -------------------------
    # CHUNKING
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    print(f"🧩 Total chunks created: {len(chunks)}")

    # -------------------------
    # CREATE VECTOR DB
    # -------------------------
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    print("✅ Vector DB initialized successfully.")

    # ❗ IMPORTANT:
    # Do NOT call db.persist()
    # Chroma v0.4+ auto-persists