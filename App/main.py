from data_loader import load_data
from vector_store import index_data
from retriever import retrieve
from llm_engine import generate_answer
import json

def setup():
    print("Setting up system...")
    data = load_data()
    index_data(data)
    print("Setup complete.")

def run_query(input_data) -> str:
    try:
        history = input_data.get("history", [])
        query = input_data.get("current_input", "")

        mode = "final" if "generate srs" in query.lower() else "clarify"

        rag_output = retrieve(query)

        response = generate_answer(
            query=query,
            rag_output=rag_output,
            history=history,
            mode=mode
        )

        return response

    except Exception as e:
        return json.dumps({
            "inference": "System error",
            "cqs": [],
            "ambiguity_score": 10,
            "reasoning": str(e),
            "confidence_score": 0.5
        })


if __name__ == "__main__":
    setup()

    while True:
        q = input("Ask: ")
        print(run_query({"history": [], "current_input": q}))