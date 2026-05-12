import os
import shutil

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _get_ffmpeg_location():
    """Return an ffmpeg location yt-dlp can use, or raise a helpful error."""
    configured = os.getenv("FFMPEG_LOCATION")
    if configured:
        if os.path.isdir(configured):
            ffmpeg_path = os.path.join(configured, "ffmpeg")
            ffprobe_path = os.path.join(configured, "ffprobe")
            if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
                return configured
        elif os.path.basename(configured) == "ffmpeg" and os.path.exists(configured):
            sibling_ffprobe = os.path.join(os.path.dirname(configured), "ffprobe")
            if os.path.exists(sibling_ffprobe):
                return os.path.dirname(configured)

        raise RuntimeError(
            "FFMPEG_LOCATION is set, but both ffmpeg and ffprobe were not found there."
        )

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path and ffprobe_path:
        return os.path.dirname(ffmpeg_path)

    raise RuntimeError(
        "ffmpeg and ffprobe are required for YouTube audio extraction. "
        "Install them with `brew install ffmpeg` on macOS, or set "
        "the FFMPEG_LOCATION environment variable to the directory that contains both binaries."
    )


def download_audio_from_youtube(url):
    ffmpeg_location = _get_ffmpeg_location()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"),
        "ffmpeg_location": ffmpeg_location,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_file = (
            ydl.prepare_filename(info_dict).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        )
    return audio_file

def convert_to_wav(input_path: str) -> str:
    """Convert an audio file to WAV format."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1) #16kHz mono is ideal for speech recognition
    
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_length_ms: int = 600000) -> list[str]:
    """Split an audio file into chunks of a specified length."""
    audio = AudioSegment.from_file(wav_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"{os.path.splitext(wav_path)[0]}_chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(input_source: str) -> list[str]:
    """Given an input source (YouTube URL or local file), return a list of WAV chunk paths."""
    if input_source.startswith("http") or input_source.startswith("https"):
        print(f"Downloading audio from YouTube URL ..........")
        wav_path = download_audio_from_youtube(input_source)
    else:
        print(f"Detected local file. Processing ..........")
        wav_path = convert_to_wav(input_source)

    print(f"Chunking audio into 10-minute segments ..........")

    chunk_paths = chunk_audio(wav_path)
    print(f"Audio processing complete. Generated {len(chunk_paths)} chunk(s).")
    return chunk_paths