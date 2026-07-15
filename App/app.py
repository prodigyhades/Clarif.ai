import streamlit as st
import json
from main import run_query, setup

# -------------------------
# INIT BACKEND (RUN ONCE)
# -------------------------
if "initialized" not in st.session_state:
    setup()
    st.session_state.initialized = True

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Clarif.Ai",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_display" not in st.session_state:
    st.session_state.chat_display = [
        {
            "role": "assistant",
            "content": "Hello! Enter your requirement to begin."
        }
    ]

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

if "srs_output" not in st.session_state:
    st.session_state.srs_output = None

# -------------------------
# PROCESS USER INPUT
# -------------------------
def process_user_input(user_text):
    # Add user message
    st.session_state.chat_display.append({"role": "user", "content": user_text})
    st.session_state.history.append(user_text)

    input_data = {
        "history": st.session_state.history,
        "current_input": user_text
    }

    response_raw = run_query(input_data)

    # DEBUG (IMPORTANT)
    # st.write("RAW RESPONSE:", response_raw)

    try:
        response_data = json.loads(response_raw)

        # -------------------------
        # FINAL SRS
        # -------------------------
        if "final_srs" in response_data:
            st.session_state.srs_output = response_data["final_srs"]

            st.session_state.chat_display.append({
                "role": "assistant",
                "content": "✅ SRS generated successfully!"
            })

            st.session_state.current_analysis = None

        # -------------------------
        # CLARIFICATION MODE
        # -------------------------
        else:
            st.session_state.current_analysis = response_data

            st.session_state.chat_display.append({
                "role": "assistant",
                "content": "I've analyzed your input. Please review the dashboard below."
            })

            # store structured response (important for multi-turn)
            st.session_state.history.append(json.dumps(response_data))

    except Exception as e:
        st.error("❌ Backend returned invalid JSON")
        st.write("Error:", e)
        st.write("Raw Response:", response_raw)

    st.rerun()

# -------------------------
# UI HEADER
# -------------------------
st.title("🧠 Clarif.Ai - AI Requirement Clarifier")
st.markdown("Convert vague ideas into structured specifications.")
st.divider()

# -------------------------
# CHAT UI
# -------------------------
st.subheader("💬 Conversation")

with st.container():
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# -------------------------
# INPUT BOX
# -------------------------
if st.session_state.current_analysis is None or not st.session_state.current_analysis.get("cqs"):
    user_prompt = st.chat_input("Enter your requirement...")

    if user_prompt:
        process_user_input(user_prompt)

# -------------------------
# ANALYSIS DASHBOARD
# -------------------------
if st.session_state.current_analysis:

    analysis = st.session_state.current_analysis

    score = analysis.get("ambiguity_score", 10)
    confidence = int(analysis.get("confidence_score", 0) * 100)

    st.divider()
    st.subheader("📊 Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    # -------- Inference --------
    with col1:
        st.markdown("### 🧾 Inference")
        st.info(analysis.get("inference", ""))

        with st.expander("Reasoning"):
            st.write(analysis.get("reasoning", ""))

    # -------- Ambiguity --------
    with col2:
        st.markdown("### 🎯 Ambiguity Score")
        st.metric("Score", score)

    # -------- Confidence --------
    with col3:
        st.markdown("### 🤖 Confidence")
        st.metric("Confidence", f"{confidence}%")

    # -------------------------
    # CLARIFYING QUESTIONS
    # -------------------------
    if analysis.get("cqs"):

        st.subheader("❓ Clarifying Questions")

        with st.form("questions_form"):
            answers = {}

            for i, q in enumerate(analysis["cqs"]):
                answers[q] = st.text_area(q, key=f"q_{i}")

            submit = st.form_submit_button("Submit Answers")

            if submit:
                combined = "\n".join(
                    [f"{q}: {a if a else 'Not provided'}" for q, a in answers.items()]
                )

                process_user_input(combined)

    # -------------------------
    # GENERATE SRS BUTTON
    # -------------------------
    if score <= 3:
        st.success("Requirements are clear enough to generate SRS.")

        if st.button("📄 Generate SRS"):
            process_user_input("Generate SRS")

# -------------------------
# SRS OUTPUT
# -------------------------
if st.session_state.srs_output:

    st.divider()
    st.subheader("📑 Final SRS Document")

    st.markdown(st.session_state.srs_output)

    st.download_button(
        label="⬇️ Download SRS",
        data=st.session_state.srs_output,
        file_name="srs.md",
        mime="text/markdown"
    )