import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize_transcript, generate_title
from core.extracter import (
    extract_actionable_items,
    extract_key_decisions,
    extract_key_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #f8fafc;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 20px;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    border: 1px solid #334155;
}

.metric-card {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.chat-user {
    background-color: #2563eb;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.chat-bot {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="big-title">🎥 AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Transcribe, summarize, extract insights, and chat with your meetings/videos.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    input_mode = st.radio(
        "Choose Input Type",
        ["YouTube URL", "Upload Audio/Video File"]
    )

    language = st.selectbox(
        "Language",
        ["english", "hinglish"]
    )

# -----------------------------
# Input Section
# -----------------------------
source = None

if input_mode == "YouTube URL":
    source = st.text_input("🔗 Enter YouTube URL")

else:
    uploaded_file = st.file_uploader(
        "📁 Upload Audio/Video",
        type=["mp3", "wav", "mp4", "m4a"]
    )

    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        source = temp_path

# -----------------------------
# Run Pipeline
# -----------------------------
if st.button("🚀 Process Content", use_container_width=True):

    if not source:
        st.warning("Please provide a valid input source.")
        st.stop()

    with st.spinner("Processing your content..."):

        # Step 1
        chunks = process_input(source)

        # Step 2
        transcript = transcribe_all(chunks)

        # Step 3
        title = generate_title(transcript)
        summary = summarize_transcript(transcript)

        # Step 4
        actionable_items = extract_actionable_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_key_questions(transcript)

        # Step 5
        rag_chain = build_rag_chain(transcript)

        st.session_state.rag_chain = rag_chain
        st.session_state.transcript = transcript

    st.success("✅ Processing Complete!")

    # -----------------------------
    # Dashboard Metrics
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 Transcript</h3>
            <h2>{len(transcript.split())} words</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Action Items</h3>
            <h2>{len(str(actionable_items).splitlines())}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>❓ Questions</h3>
            <h2>{len(str(questions).splitlines())}</h2>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # Title
    # -----------------------------
    st.markdown(f"""
    <div class="card">
        <h2>📌 {title}</h2>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary",
        "✅ Action Items",
        "🔑 Decisions",
        "❓ Questions",
        "📜 Transcript"
    ])

    with tab1:
        st.markdown(summary)

    with tab2:
        st.markdown(actionable_items)

    with tab3:
        st.markdown(decisions)

    with tab4:
        st.markdown(questions)

    with tab5:
        st.text_area(
            "Transcript",
            transcript,
            height=400
        )

# -----------------------------
# Chat Section
# -----------------------------
if st.session_state.rag_chain:

    st.divider()
    st.subheader("💬 Chat With Your Meeting")

    user_question = st.chat_input("Ask anything about the transcript...")

    if user_question:

        st.session_state.chat_history.append(("user", user_question))

        with st.spinner("Thinking..."):
            answer = ask_question(
                st.session_state.rag_chain,
                user_question
            )

        st.session_state.chat_history.append(("assistant", answer))

    # Display Chat History
    for role, message in st.session_state.chat_history:

        if role == "user":
            st.markdown(
                f'<div class="chat-user"><b>You:</b> {message}</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="chat-bot"><b>Assistant:</b> {message}</div>',
                unsafe_allow_html=True
            )