from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize_transcript, generate_title
from core.extracter import extract_actionable_items, extract_key_decisions, extract_key_questions
from core.rag_engine import build_rag_chain, ask_question
load_dotenv()

def run_pipeline(source: str, language : str = "english") -> dict:

    print("Starting AI video Assistant...")
    # Step 1: Process the input audio
    chunks = process_input(source)
    
    # Step 2: Transcribe the audio chunks
    transcript = transcribe_all(chunks)
    print(f"Raw Transcript(First 300 characters): {transcript[:300]}...")  # Print the first 300 characters of the transcript for verification
    
    # Step 3: Generate summary and title
    title = generate_title(transcript)
    summary = summarize_transcript(transcript)
    
    # Step 4: Extract actionable items, decisions, and questions
    actionable_items = extract_actionable_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_key_questions(transcript)

    # Step 5: Build RAG chain for question answering
    rag_chain = build_rag_chain(transcript)

    # Return the results
    return {
        "title": title,
        "summary": summary,
        "actionable_items": actionable_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['actionable_items']}")
    print(f"\n🔑 Key Decisions:\n{result['decisions']}")
    print(f"\n❓ Open Questions:\n{result['questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")