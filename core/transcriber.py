import os
import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # Default to "small" if not set

_model = None
def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model '{WHISPER_MODEL}'...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Model loaded successfully.")
    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    """Transcribe an audio file using the Whisper model."""
    model = load_model()
    task = "translate" if translate else "transcribe"
    results = model.transcribe(chunk_path, task=task)
    return results["text"]  

def transcribe_all(chunks:list,translate: bool = False) -> str:
    """Transcribe a list of audio chunks and concatenate the results."""
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk: {i + 1}...")
        transcript = transcribe_chunk(chunk, translate=translate)
        full_transcript += transcript + " "
        print(f"Chunk {i + 1} transcription complete.") 
    return full_transcript