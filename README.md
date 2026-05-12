
# 🎙️ AI Meeting Assistant

An AI-powered Meeting Assistant that can process meeting recordings or YouTube videos, generate transcripts, summarize discussions, extract actionable insights, and even allow users to chat with their meetings using RAG (Retrieval-Augmented Generation).



#  Features

🎧 Process local audio/video files <br>
📺 Process YouTube meeting recordings <br>
📝 Automatic speech-to-text transcription <br>
🧠 AI-generated meeting summaries <br>
📌 Smart meeting title generation <br>
✅ Extract actionable tasks <br>
🔑 Detect key decisions <br>
❓ Extract important/open questions <br>
💬 Chat with your meeting using RAG <br>
🌐 Supports English & Hinglish inputs <br>




# Project Architecture



```
Input Audio / YouTube URL
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
     Interactive Q&A Chat
```



# Project Structure

```
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

```
