# AI-Meeting-Assistant
you can ask question from meeting transcript ,gives u clear idea and future goal


# 🎙️ AI Meeting Assistant

An AI-powered Meeting Assistant that can process meeting recordings or YouTube videos, generate transcripts, summarize discussions, extract actionable insights, and even allow users to chat with their meetings using RAG (Retrieval-Augmented Generation).

Built using modern Generative AI workflows, speech recognition, and LLM-powered analysis

✨ Features
🎧 Process local audio/video files
📺 Process YouTube meeting recordings
📝 Automatic speech-to-text transcription
🧠 AI-generated meeting summaries
📌 Smart meeting title generation
✅ Extract actionable tasks
🔑 Detect key decisions
❓ Extract important/open questions
💬 Chat with your meeting using RAG
🌐 Supports English & Hinglish inputs


# Project Architecture

'''Input Audio / YouTube URL
            │
            ▼
   Audio Processing Layer
            │
            ▼
      Speech Transcription
            │
            ▼
      LLM-Based Analysis
   ├── Summary Generation
   ├── Title Generation
   ├── Action Items
   ├── Key Decisions
   └── Open Questions
            │
            ▼
      RAG Knowledge Base
            │
            ▼
     Interactive Q&A Chat'''




# Project Structure

     AI-Meeting-Assistant/
│
├── core/
│   ├── transcriber.py
│   ├── summarize.py
│   ├── extracter.py
│   └── rag_engine.py
│
├── utils/
│   └── audio_processor.py
│
├── main.py
├── requirements.txt
├── .env
└── README.md
