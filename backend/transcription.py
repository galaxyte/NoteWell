"""
Video-to-Notes AI Platform — STEP 2: Whisper fallback
Used only when a video has NO captions (youtube-transcript-api came up empty).

Pipeline (Section 4, step 4 of the plan):
  download audio only with yt-dlp -> transcribe.

Transcription runs locally with faster-whisper. Audio is never uploaded to an
external transcription service.
"""

import os
import tempfile
import shutil
from dataclasses import dataclass

from fastapi import HTTPException
from typing import Optional, Callable

# faster-whisper loads its model lazily and reuses it across requests instead of
# reloading (which is slow) on every single video.
_whisper_model = None

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # tiny/base/small/medium
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 = fastest on CPU


@dataclass
class WhisperResult:
    segments: list       # [{"start": float, "end": float, "text": str}, ...]
    language: str
    duration_seconds: float

    @property
    def text(self) -> str:
        return " ".join(seg["text"].strip() for seg in self.segments)


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "faster-whisper is not installed. Run "
                    "'pip install -r requirements.txt' inside backend/ and try again."
                ),
            ) from exc

        # Downloads the model from Hugging Face on first use only, then caches it locally.
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _whisper_model


def download_audio(video_url: str, workdir: str) -> str:
    """Download ONLY the audio track (not the full video) for a YouTube URL via yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "yt-dlp is not installed. Run 'pip install -r requirements.txt' "
                "inside backend/ and try again."
            ),
        ) from exc

    attempts = [
        {
            "format": "bestaudio/best",
            "note": "audio-only",
        },
        {
            "format": "best[acodec!=none]/best",
            "extractor_args": {"youtube": {"player_client": ["android"]}},
            "note": "android progressive fallback",
        },
    ]

    last_error = None
    for attempt in attempts:
        for name in os.listdir(workdir):
            if name.startswith("audio."):
                os.remove(os.path.join(workdir, name))

        ydl_opts = {
            "format": attempt["format"],
            "outtmpl": os.path.join(workdir, "audio.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if "extractor_args" in attempt:
            ydl_opts["extractor_args"] = attempt["extractor_args"]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(video_url, download=True)
            break
        except Exception as exc:  # yt-dlp raises its own broad DownloadError
            last_error = exc
            print(f"yt-dlp {attempt['note']} download failed: {exc}")
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Could not download audio for this video: {last_error}",
        ) from last_error

    # yt-dlp picks the final extension based on what the site served (m4a/webm/etc.)
    downloaded = [f for f in os.listdir(workdir) if f.startswith("audio.")]
    if not downloaded:
        raise HTTPException(status_code=422, detail="Audio download produced no file.")
    return os.path.join(workdir, downloaded[0])


def transcribe_audio(
    audio_path: str,
    start_offset: float = 0.0,
    progress_callback: Optional[Callable[[float, float], None]] = None,
) -> WhisperResult:
    """
    progress_callback(processed_seconds, total_seconds), if given, is called
    once immediately with (0, total) -- faster-whisper knows total duration
    from file metadata before transcribing anything -- and again after every
    segment, so the caller gets live progress instead of only a final result.
    """
    model = _get_model()
    segments_iter, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    total_seconds = info.duration
    if progress_callback is not None:
        progress_callback(0.0, total_seconds)

    segments = []
    for seg in segments_iter:
        entry = {"start": seg.start + start_offset, "end": seg.end + start_offset, "text": seg.text.strip()}
        segments.append(entry)
        if progress_callback is not None:
            progress_callback(entry["end"], total_seconds)

    return WhisperResult(segments=segments, language=info.language, duration_seconds=info.duration)


def trim_audio(input_path: str, start_seconds: float, end_seconds: float, workdir: str) -> str:
    """Slice [start_seconds, end_seconds] out of input_path using ffmpeg (stream copy, no re-encode)."""
    import subprocess
    output_path = os.path.join(workdir, "trimmed_audio" + os.path.splitext(input_path)[1])
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ss", str(start_seconds),
           "-to", str(end_seconds), "-c", "copy", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg trim failed: {result.stderr}")
    return output_path


def transcribe_video_with_whisper(
    video_url: str,
    start_seconds: float = None,
    end_seconds: float = None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
) -> WhisperResult:
    workdir = tempfile.mkdtemp(prefix="v2n_audio_")
    try:
        audio_path = download_audio(video_url, workdir)
        offset = 0.0
        if start_seconds is not None and end_seconds is not None:
            audio_path = trim_audio(audio_path, start_seconds, end_seconds, workdir)
            offset = start_seconds
        return transcribe_audio(audio_path, start_offset=offset, progress_callback=progress_callback)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
