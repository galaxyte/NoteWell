"""
Video-to-Notes AI Platform — backend API

Pipeline: any source (YouTube link, pasted text, PDF, DOCX, audio upload) is
reduced to plain text, chunked, summarized per-chunk, then combined into
structured Markdown study notes. Notes are persisted in SQLite and indexed
into a local Chroma vector store.

Features on top of that:
  * Chat — general Q&A across the whole note library, or scoped to one note.
    Both are RAG-backed. Conversations are stored as *sessions*, so past
    threads can be reopened instead of being overwritten.
  * Practice questions — generated from a saved note, in batches, with
    .docx/.pdf export that follows the answers-shown/hidden state.
  * Library — list, reopen, and delete anything previously generated.
  * Section-level edit/regenerate — rewrite or manually replace one section
    of an already-generated note without touching the rest (Phase 2).

Still not implemented: OCR for scanned/image-only PDFs, and MCP tools.
"""

import os
import re
import shutil
import tempfile
import textwrap
import threading
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

import db
import export
import questions as questions_mod
import rag
from extraction import extract_text_from_pdf, extract_text_from_docx
from learning_modes import (
    DEFAULT_LEARNING_MODE,
    apply_learning_mode,
    normalize_learning_mode,
    notes_opening_structure,
    notes_section_structure,
    notes_closing_structure,
    notes_structure,
    question_style,
)
from transcription import (
    transcribe_video_with_whisper,
    transcribe_audio,
)
from timeutils import parse_timestamp, format_timestamp, pick_interval_seconds
from datetime import datetime, timezone
from typing import List, Optional, Callable
import time

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Video-to-Notes AI")

load_dotenv()  # reads the .env file sitting next to this script (see README)
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000").split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env file and paste your Gemini "
        "API key into it before starting the server."
    )

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = FastAPI(title="Video-to-Notes AI")

# Allow the frontend to call the API even if it's opened from a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()  # creates video_to_notes.db + tables next to this file, if not already there


@app.on_event("startup")
def _backfill_rag_index() -> None:
    """Index any notes that existed before RAG was added (or were generated
    while the embedding deps weren't installed yet). Cheap no-op once
    everything is already indexed, since index_note()/upsert() is idempotent.
    Runs in the background so it never blocks the server from accepting
    requests."""

    def _run():
        try:
            result = rag.reindex_all()
            print(f"[rag] startup backfill: {result}")
        except Exception as exc:  # embedding deps missing, first-run model download, etc.
            print(f"[rag] startup backfill skipped: {exc}")

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class NotesRequest(BaseModel):
    video_url: str
    learning_mode: Optional[str] = DEFAULT_LEARNING_MODE
    start_time: Optional[str] = None   # "12:30", "1:02:00", or plain seconds; parsed below
    end_time: Optional[str] = None


class TextNotesRequest(BaseModel):
    text: str
    title: Optional[str] = None
    learning_mode: Optional[str] = DEFAULT_LEARNING_MODE


class ChatRequest(BaseModel):
    session_id: Optional[int] = None   # None = start a new conversation
    scope: str = "general"             # only read when creating a session
    message: str
    learning_mode: Optional[str] = DEFAULT_LEARNING_MODE


class QuestionsRequest(BaseModel):
    note_id: int
    count: int = 10
    learning_mode: Optional[str] = DEFAULT_LEARNING_MODE


class SectionRegenerateRequest(BaseModel):
    # Optional free-text steer, e.g. "make this shorter" / "add a concrete example".
    # None means "just rewrite it, no specific direction."
    instruction: Optional[str] = None


class SectionEditRequest(BaseModel):
    # Full replacement text for one section, heading included. Manual edit —
    # no LLM call, so unlike regenerate this CAN change the heading itself.
    markdown_text: str


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Pull the 11-character YouTube video ID out of any common URL format."""
    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&]|$|/)",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not find a valid YouTube video ID in that URL.")


def fetch_video_title(video_url: str, fallback: str) -> str:
    """Best-effort video title lookup (for the Library screen) via yt-dlp metadata
    only — no download involved. Falls back to the raw video id if this fails."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info.get("title") or fallback
    except Exception:
        return fallback


def fetch_transcript(
    video_id: str, video_url: str,
    start_seconds: Optional[float] = None, end_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
) -> tuple[List[dict], str]:
    """Try YouTube captions first (free + instant), then fall back to
    fully-local Whisper as a last resort.

    Returns (segments, origin) where segments is a list of
    {"text": str, "start": float, "duration": float} dicts (NOT a flattened
    string anymore — V1 threw this timing away, V2 needs it for clickable
    timestamps) and origin is "captions" or "whisper".

    If start_seconds/end_seconds are given, the returned segments are limited
    to that range. For captions this is a simple filter (the full transcript
    is already fetched — cheap). For both Whisper paths, the range is pushed
    down into the download/trim step so the video isn't fully transcribed and
    then discarded.
    """
    try:
        print("Trying YouTube captions...")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        segments = [
            {
                "text": item.text if hasattr(item, "text") else item["text"],
                "start": item.start if hasattr(item, "start") else item["start"],
                "duration": item.duration if hasattr(item, "duration") else item.get("duration", 0.0),
            }
            for item in transcript_list
        ]

        if start_seconds is not None and end_seconds is not None:
            segments = [s for s in segments if start_seconds <= s["start"] < end_seconds]

        if segments and any(s["text"].strip() for s in segments):
            print("Transcript fetched from YouTube captions.")
            return segments, "captions"

    except (TranscriptsDisabled, NoTranscriptFound, ET.ParseError) as e:
        print(f"YouTube captions unavailable: {e}")
    except Exception as e:
        print(f"Unexpected transcript error: {e}")

    print("Trying local Whisper...")
    result = transcribe_video_with_whisper(
        video_url, start_seconds=start_seconds, end_seconds=end_seconds,
        progress_callback=progress_callback,
    )
    origin = "whisper"

    if not result.segments:
        raise HTTPException(
            status_code=422,
            detail="Whisper transcription produced no speech text for this video.",
        )

    # Normalize to the same {"text", "start", "duration"} shape captions use
    # above -- both Whisper paths carry "end" instead of "duration" (see
    # transcription.py). Converting once here means chunk_by_time() and
    # generate_and_save() never need to know or care which source produced
    # the segments they're chunking.
    segments = [
        {"text": s["text"], "start": s["start"], "duration": s["end"] - s["start"]}
        for s in result.segments
    ]

    print(f"Transcript generated using {origin}.")
    return segments, origin

def chunk_by_time(segments: List[dict], interval_seconds: float) -> List[dict]:
    """
    Buckets timed segments (from fetch_transcript) into fixed-width time
    windows instead of fixed word counts. Returns
    [{"start": float, "end": float, "text": str}, ...] — one dict per window,
    in order, with no gaps or overlaps.

    Replaces chunk_text() for anything that has real timing (video sources).
    chunk_text() is kept as-is for the text/pdf/docx paths, which have no
    timestamps to preserve.
    """
    if not segments:
        return []

    chunks = []
    window_start = segments[0]["start"]
    window_end = window_start + interval_seconds
    current_texts: List[str] = []
    current_actual_start = segments[0]["start"]
    last_end = current_actual_start

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg_start + seg.get("duration", 0.0)

        if seg_start >= window_end and current_texts:
            chunks.append({
                "start": current_actual_start,
                "end": last_end,
                "text": " ".join(current_texts),
            })
            window_start = seg_start
            window_end = window_start + interval_seconds
            current_texts = []
            current_actual_start = seg_start

        current_texts.append(seg["text"])
        last_end = seg_end

    if current_texts:
        chunks.append({
            "start": current_actual_start,
            "end": last_end,
            "text": " ".join(current_texts),
        })

    return chunks


CHUNK_WORDS = 900  # untimed sources only (pasted text/pdf/docx) -- video/audio
                    # use chunk_by_time() instead, since those have real timing


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> List[str]:
    """
    Word-count chunking for untimed sources. No timestamps exist for pasted
    text/PDF/DOCX, so there's nothing to chunk BY except a fixed word
    window -- kept as its own simple function rather than merged into
    chunk_by_time(), since that function's whole design is built around
    seconds, which doesn't apply here at all.
    """
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), chunk_words)
    ]


def call_gemini(prompt: str, system: str, temperature: float = 0.3) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        ),
    )
    return response.text or ""


def summarize_chunk(
    chunk: str,
    index: int,
    total: int,
    learning_mode: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> str:
    """
    Map step: summarize one transcript chunk.

    start/end (seconds) are purely informational context handed to the model —
    NOT a formatting instruction. The model is told what time range this chunk
    covers so its summary can reference "early on" / "toward the end" type
    framing if natural; the actual timestamped heading in the final notes is
    assembled in Python (build_notes), not generated by the model.
    """
    time_context = ""
    if start is not None and end is not None:
        time_context = f"This section covers {format_timestamp(start)}–{format_timestamp(end)} of the source. "

    prompt = (
        f"This is part {index + 1} of {total} of a transcript. {time_context}"
        f"Summarize the key points concisely. Preserve every proper noun, tool name, version "
        f"number, and figure exactly as stated — later stages depend on "
        f"them:\n\n{chunk}"
    )
    system = (
        "You are an expert note-taker distilling transcripts into concise "
        "bullet-point summaries. You never drop a specific detail in favor of "
        "a general paraphrase."
    )
    return call_gemini(prompt, system=system)


def write_section(chunk_text: str, index: int, total: int, learning_mode: str, start: float, end: float) -> str:
    """Writes ONE final, ready-to-render H2 section directly from this chunk's
    raw text -- not an intermediate summary. Path B streams this straight to
    the user the moment it's written, so there's no later blending step that
    could fix up a rough draft; it has to be right the first time."""
    prompt = textwrap.dedent(f"""
        {notes_section_structure(learning_mode)}

        This is part {index + 1} of {total} of the source. Use exactly this
        bracketed time range in your heading: [{format_timestamp(start)}\u2013{format_timestamp(end)}]

        Source text for this section:
        {chunk_text}
    """).strip()
    system = apply_learning_mode(
        "You write one final, polished section of Markdown study notes -- not a summary, not a draft.",
        learning_mode,
    )
    return call_gemini(prompt, system=system)


def write_opening(sections_markdown: str, learning_mode: str) -> str:
    prompt = textwrap.dedent(f"""
        {notes_opening_structure(learning_mode)}

        Finished sections:
        {sections_markdown}
    """).strip()
    system = apply_learning_mode(
        "You write the opening of a set of Markdown study notes, given the notes' already-finished sections.",
        learning_mode,
    )
    return call_gemini(prompt, system=system)


def write_closing(sections_markdown: str, learning_mode: str) -> str:
    prompt = textwrap.dedent(f"""
        {notes_closing_structure(learning_mode)}

        Finished sections:
        {sections_markdown}
    """).strip()
    system = apply_learning_mode(
        "You write the closing of a set of Markdown study notes, given the notes' already-finished sections.",
        learning_mode,
    )
    return call_gemini(prompt, system=system)

def _run_video_generation_job(
    *,
    job_id: int,
    video_id: str,
    video_url: str,
    title: str,
    source_type: str,
    learning_mode: str,
    start_seconds: Optional[float],
    end_seconds: Optional[float],
) -> None:
    """
    Runs the full video/audio notes pipeline in a background thread. Streams
    each section into generation_job_sections the moment it's written (Path
    B), so the frontend's poll can render real content incrementally instead
    of waiting for the whole note. Also reports live transcription progress
    via a throttled callback passed into fetch_transcript() -- only actually
    incremental when the local-Whisper path is the one that ends up running;
    progress.
    """
    try:
        last_report_time = [0.0]   # mutable holder so the closure can update it

        def _report_progress(processed: float, total: float) -> None:
            now = time.monotonic()
            # Throttle: faster-whisper can yield many segments per second
            # during short/silent stretches -- writing to SQLite on every
            # single one would be wasteful. Once every ~2s of wall-clock
            # time is plenty for a progress bar a human is glancing at.
            if now - last_report_time[0] < 2.0 and processed < total:
                return
            last_report_time[0] = now
            db.update_job_progress(job_id, processed_seconds=processed, total_duration_seconds=total)

        segments, origin = fetch_transcript(
            video_id, video_url, start_seconds, end_seconds, progress_callback=_report_progress
        )

        if not segments:
            db.mark_job_failed(job_id, "No transcript content found to summarize.")
            return

        covered_start = segments[0]["start"]
        covered_end = segments[-1]["start"] + segments[-1]["duration"]
        total_duration = covered_end - covered_start
        interval_seconds = pick_interval_seconds(total_duration)
        chunks = chunk_by_time(segments, interval_seconds)

        db.update_job_progress(
            job_id,
            stage="writing_sections",
            total_duration_seconds=total_duration,
            processed_seconds=total_duration,   # transcription is fully done by this point
            total_chunks=len(chunks),
        )

        section_texts = []
        for i, c in enumerate(chunks):
            section_md = write_section(c["text"], i, len(chunks), learning_mode, c["start"], c["end"])
            section_texts.append(section_md)
            db.add_job_section(job_id, i, c["start"], c["end"], section_md)
            db.update_job_progress(job_id, completed_chunks=i + 1)

        db.update_job_progress(job_id, stage="finalizing")

        sections_combined = "\n\n".join(section_texts)
        opening_md = write_opening(sections_combined, learning_mode)
        closing_md = write_closing(sections_combined, learning_mode)
        notes_markdown = f"{opening_md}\n\n{sections_combined}\n\n{closing_md}"

        note_id = db.save_note(
            title=title,
            source_url=video_url,
            video_id=video_id,
            markdown=notes_markdown,
            num_chunks=len(chunks),
            transcript_origin=origin,
            source_type=source_type,
            learning_mode=learning_mode,
            range_start_seconds=segments[0]["start"],
            range_end_seconds=segments[-1]["start"] + segments[-1]["duration"],
        )

        timed_sections = export.extract_timed_sections(notes_markdown)
        db.save_note_sections(note_id, timed_sections)

        try:
            rag.reindex_note_clean(note_id, title, notes_markdown)
        except Exception as exc:
            print(f"[rag] failed to index note {note_id}: {exc}")

        db.mark_job_done(job_id, note_id)

    except Exception as exc:
        print(f"[job {job_id}] failed: {exc}")
        db.mark_job_failed(job_id, str(exc))

def build_notes(chunk_summaries: List[str], learning_mode: str) -> str:
    """Reduce step: combine chunk summaries into final structured Markdown notes.

    The output structure comes from learning_modes, not from here — an
    identical template across all three modes was what made them
    indistinguishable.
    """
    combined = "\n\n".join(chunk_summaries)
    prompt = textwrap.dedent(f"""
        Using the following chunk summaries from a source, produce study notes.

        {notes_structure(learning_mode)}

        Preserve every specific detail the summaries contain — names,
        versions, numbers, tools. Do not restate filler.

        Chunk summaries:
        {combined}
    """).strip()
    system = apply_learning_mode(
        "You produce clean, well-structured Markdown study notes.",
        learning_mode,
    )
    return call_gemini(prompt, system=system)


# ---------------------------------------------------------------------------
# Shared pipeline: any source type ends up here once reduced to plain text
# ---------------------------------------------------------------------------

def generate_and_save(
    *,
    text: Optional[str] = None,
    segments: Optional[List[dict]] = None,
    title: str,
    source_url: str,
    video_id: Optional[str],
    source_type: str,
    origin: str,
    learning_mode: Optional[str] = None,
) -> dict:
    """
    Exactly one of `text` or `segments` is given:
      - `segments` (video/audio sources): timed [{"text","start","duration"}] from
        fetch_transcript — chunked by TIME so notes get real timestamped headings.
      - `text` (pasted text/pdf/docx): no timing exists for these sources, so they
        keep the original word-count chunking. Nothing to timestamp here.
    """
    learning_mode = normalize_learning_mode(learning_mode)

    if segments is not None:
        if not segments:
            raise HTTPException(status_code=422, detail="No transcript content found to summarize.")

        covered_start = segments[0]["start"]
        covered_end = segments[-1]["start"] + segments[-1]["duration"]
        interval_seconds = pick_interval_seconds(covered_end - covered_start)

        chunks = chunk_by_time(segments, interval_seconds)
        chunk_summaries = [
            # Prefix with the time label the reduce step needs to build headings from —
            # the model isn't asked to invent this, it's handed the real computed range.
            f"[{format_timestamp(c['start'])}\u2013{format_timestamp(c['end'])}] "
            + summarize_chunk(c["text"], i, len(chunks), learning_mode, start=c["start"], end=c["end"])
            for i, c in enumerate(chunks)
        ]
    else:
        if not text or not text.strip():
            raise HTTPException(status_code=422, detail="No text content found to summarize.")

        chunks = chunk_text(text)
        chunk_summaries = [
            summarize_chunk(c, i, len(chunks), learning_mode) for i, c in enumerate(chunks)
        ]

    notes_markdown = build_notes(chunk_summaries, learning_mode)

    # Reuse the range already computed for chunking above (covered_start/
    # covered_end only exist inside the `if segments is not None` branch, so
    # this is where they collapse to None for the text/pdf/docx path).
    range_start_seconds = segments[0]["start"] if segments is not None else None
    range_end_seconds = (
        segments[-1]["start"] + segments[-1]["duration"] if segments is not None else None
    )

    note_id = db.save_note(
        title=title,
        source_url=source_url,
        video_id=video_id,
        markdown=notes_markdown,
        num_chunks=len(chunks),
        transcript_origin=origin,
        source_type=source_type,
        learning_mode=learning_mode,
        range_start_seconds=range_start_seconds,
        range_end_seconds=range_end_seconds,
    )

    # Pull every "[MM:SS–MM:SS] Title" heading the model produced into
    # structured rows, so the frontend can build a clickable timeline without
    # regex-parsing markdown itself. No-op (empty list) for untimed sources —
    # extract_timed_sections() only matches that exact bracketed shape.
    timed_sections = export.extract_timed_sections(notes_markdown)
    db.save_note_sections(note_id, timed_sections)

    try:
        rag.reindex_note_clean(note_id, title, notes_markdown)
    except Exception as exc:
        print(f"[rag] failed to index note {note_id}: {exc}")

    return {
        "note_id": note_id,
        "video_id": video_id,
        "title": title,
        "num_chunks": len(chunks),
        "transcript_origin": origin,
        "source_type": source_type,
        "notes_markdown": notes_markdown,
        "learning_mode": learning_mode,
    }


# ---------------------------------------------------------------------------
# Notes generation routes
# ---------------------------------------------------------------------------

@app.post("/api/generate-notes")
def generate_notes(req: NotesRequest):
    """Source: pasted YouTube link. Returns a job_id immediately -- the
    actual pipeline runs in a background thread. Poll GET /api/jobs/{job_id}
    for progress, streamed sections, and the final note_id."""
    try:
        video_id = extract_video_id(req.video_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    start_seconds = end_seconds = None
    if req.start_time is not None:
        try:
            start_seconds = parse_timestamp(req.start_time)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if req.end_time is not None:
        try:
            end_seconds = parse_timestamp(req.end_time)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if start_seconds is not None and end_seconds is not None and start_seconds >= end_seconds:
        raise HTTPException(status_code=400, detail="start_time must be before end_time.")

    title = fetch_video_title(req.video_url, fallback=video_id)
    learning_mode = normalize_learning_mode(req.learning_mode)

    job_id = db.create_generation_job(total_duration_seconds=None)

    threading.Thread(
        target=_run_video_generation_job,
        kwargs=dict(
            job_id=job_id, video_id=video_id, video_url=req.video_url, title=title,
            source_type="video", learning_mode=learning_mode,
            start_seconds=start_seconds, end_seconds=end_seconds,
        ),
        daemon=True,
    ).start()

    return {"job_id": job_id}

def _estimate_seconds_remaining(job: dict) -> Optional[int]:
    """
    Computed fresh on every poll, never stored (it's derived). Two different
    units depending on stage, so two different formulas:

    - "transcribing": ratio of audio processed vs total. Only meaningful once
      transcription.py reports processed_seconds incrementally (not yet true
      -- see the caveat in _run_video_generation_job). Until then, falls back
      to a rough guess (assume ~real-time speed), clearly not a measurement.
    - "writing_sections": ratio of chunks completed vs total, using
      stage_started_at so time spent transcribing doesn't skew the rate.
    - anything else: no estimate.
    """
    if job["status"] != "running":
        return None
    stage_started = job.get("stage_started_at") or job.get("started_at")
    if not stage_started:
        return None
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(stage_started)).total_seconds()

    if job["stage"] == "transcribing":
        total = job.get("total_duration_seconds")
        processed = job.get("processed_seconds") or 0
        if not total:
            return None
        if processed <= 0:
            return int(total)   # rough guess only -- no live signal yet
        rate = elapsed / processed
        return round(rate * max(total - processed, 0))

    if job["stage"] == "writing_sections":
        total_chunks = job.get("total_chunks") or 0
        completed = job.get("completed_chunks") or 0
        if not total_chunks or completed <= 0:
            return None
        rate = elapsed / completed
        return round(rate * max(total_chunks - completed, 0))

    return None


@app.get("/api/jobs/{job_id}")
def get_generation_job(job_id: int):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    job["estimated_seconds_remaining"] = _estimate_seconds_remaining(job)
    return job

@app.post("/api/generate-notes/text")
def generate_notes_from_text(req: TextNotesRequest):
    """Source: pasted plain text — no extraction needed, straight into the pipeline."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Paste some text first.")

    title = (req.title or "").strip()
    if not title:
        title = text[:60].strip() + ("…" if len(text) > 60 else "")

    return generate_and_save(
        text=text,
        title=title,
        source_url="",
        video_id=None,
        source_type="text",
        origin="pasted_text",
        learning_mode=req.learning_mode,
    )


@app.post("/api/generate-notes/pdf")
async def generate_notes_from_pdf(
    file: UploadFile = File(...),
    learning_mode: str = Form(DEFAULT_LEARNING_MODE),
):
    """Source: uploaded PDF — text extracted directly, no OCR (scanned PDFs won't work)."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Couldn't extract any text from that PDF — it may be scanned/image-only.",
        )

    title = os.path.splitext(file.filename)[0]
    return generate_and_save(
        text=text,
        title=title,
        source_url=file.filename,
        video_id=None,
        source_type="pdf",
        origin="pdf_extraction",
        learning_mode=learning_mode,
    )


@app.post("/api/generate-notes/docx")
async def generate_notes_from_docx(
    file: UploadFile = File(...),
    learning_mode: str = Form(DEFAULT_LEARNING_MODE),
):
    """Source: uploaded Word document — plain-paragraph text extraction."""
    if not (file.filename or "").lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Please upload a .docx file.")

    file_bytes = await file.read()
    text = extract_text_from_docx(file_bytes)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Couldn't extract any text from that document.")

    title = os.path.splitext(file.filename)[0]
    return generate_and_save(
        text=text,
        title=title,
        source_url=file.filename,
        video_id=None,
        source_type="docx",
        origin="docx_extraction",
        learning_mode=learning_mode,
    )


@app.post("/api/generate-notes/audio")
async def generate_notes_from_audio(
    file: UploadFile = File(...),
    learning_mode: str = Form(DEFAULT_LEARNING_MODE),
):
    """Source: uploaded audio file — transcribed locally with the same faster-whisper
    model used as the YouTube-captions fallback."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="That audio file came through empty.")

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    workdir = tempfile.mkdtemp(prefix="v2n_upload_")
    try:
        audio_path = os.path.join(workdir, f"upload{suffix}")
        with open(audio_path, "wb") as f:
            f.write(file_bytes)
        result = transcribe_audio(audio_path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if not result.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Whisper produced no speech text for this audio file.",
        )

    title = os.path.splitext(file.filename or "Audio note")[0]
    return generate_and_save(
        text=result.text,
        title=title,
        source_url=file.filename or "",
        video_id=None,
        source_type="audio",
        origin="whisper",
        learning_mode=learning_mode,
    )


@app.get("/api/notes/{note_id}/sections")
def get_note_sections(note_id: int):
    """Structured timeline data for one note — [{"order_index","start_seconds",
    "end_seconds","heading"}, ...]. Empty list for untimed sources (text/pdf/docx)
    or notes generated before this feature existed."""
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"sections": db.get_note_sections(note_id)}


# ---------------------------------------------------------------------------
# Section-level edit & regenerate (Phase 2)
#
# Distinct from GET /api/notes/{note_id}/sections above: that endpoint reads
# from the note_sections TABLE (timestamp + heading only, built for the
# seek-chip timeline in Phase 1). This one reads the note's live markdown and
# splits it fresh via export.split_into_sections(), because edit/regenerate
# need each section's actual body text, not just its timing metadata.
# ---------------------------------------------------------------------------

@app.get("/api/notes/{note_id}/editable-sections")
def get_editable_sections(note_id: int):
    """Full section list for the edit/regenerate UI:
    [{"index","heading","markdown_text"}, ...]. Recomputed on every call from
    the note's current structured_markdown (cheap, pure-Python split — no
    need to persist this separately from the source of truth)."""
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"sections": export.split_into_sections(note["structured_markdown"])}


def _rebuild_note_markdown(sections: List[dict]) -> str:
    """Joins a section list (from export.split_into_sections(), one entry
    possibly just replaced) back into one markdown string. Sections are
    already newline-separated internally; a blank line between them keeps
    Markdown block boundaries (headings, lists) intact."""
    return "\n\n".join(s["markdown_text"] for s in sections)


def _save_edited_note(note_id: int, title: str, sections: List[dict]) -> str:
    """
    Shared tail end of both the regenerate and manual-edit endpoints below.
    Splices the section list into one markdown string, saves it, and
    re-derives everything downstream that depends on note content:

      1. db.update_note_markdown()   -- the note itself
      2. export.extract_timed_sections() + db.save_note_sections()
         -- re-parses the (possibly-changed) headings for the Phase 1
            seek-chip timeline. A shortened/lengthened section can shift
            which headings still match the [MM:SS–MM:SS] pattern, so this
            has to re-run on every save, not just at generation time.
      3. rag.reindex_note_clean()    -- the RAG index (see bug 3.1: a plain
         index_note() upsert would leave stale chunks behind if the note
         shrinks; reindex_note_clean() deletes first)

    These are the exact same three calls generate_and_save() already makes
    for a brand-new note -- edit/regenerate just re-runs them against an
    updated markdown string instead of a freshly-generated one.
    """
    new_markdown = _rebuild_note_markdown(sections)
    db.update_note_markdown(note_id, new_markdown)

    timed_sections = export.extract_timed_sections(new_markdown)
    db.save_note_sections(note_id, timed_sections)

    try:
        rag.reindex_note_clean(note_id, title, new_markdown)
    except Exception as exc:
        print(f"[rag] failed to reindex note {note_id} after edit: {exc}")

    return new_markdown


@app.post("/api/notes/{note_id}/sections/{index}/regenerate")
def regenerate_section(note_id: int, index: int, req: SectionRegenerateRequest):
    """
    Scoped LLM regenerate of ONE section -- one Gemini call, not a full-note
    regeneration. The heading line (including any [MM:SS–MM:SS] bracket) is
    split off and never sent to the model for rewriting; only the body is
    regenerated, then reassembled as `heading + new_body`. This guarantees a
    regenerate can never silently drop or reformat the timestamp bracket a
    Phase-1 seek chip depends on -- the model literally never sees it.
    """
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")

    sections = export.split_into_sections(note["structured_markdown"])
    if index < 0 or index >= len(sections):
        raise HTTPException(status_code=404, detail="Section index out of range.")

    target_lines = sections[index]["markdown_text"].splitlines()
    heading_line = target_lines[0] if target_lines else f"## {sections[index]['heading']}"
    current_body = "\n".join(target_lines[1:]).strip()

    # Continuity context: the note's opening section stands in for a
    # dedicated TL;DR field (none is stored separately). Skipped when
    # regenerating the opening itself, so it isn't handed its own current
    # text as "context to stay consistent with."
    context = sections[0]["markdown_text"] if index != 0 and sections else ""

    instruction_block = (
        f"\n\nInstruction from the user for this rewrite: {req.instruction}"
        if req.instruction else ""
    )
    context_block = (
        f"Overall notes context, for consistency only -- do not repeat this back:\n{context}\n\n"
        if context else ""
    )

    prompt = textwrap.dedent(f"""
        Rewrite ONLY the body text of one section from a set of study notes.
        Do not write a heading -- the heading is fixed and handled separately,
        outside what you're asked to produce. Keep the same topic/scope as the
        current body; don't wander into material that belongs in a different
        section.

        {context_block}Current body text for this section:
        {current_body}
        {instruction_block}
    """).strip()
    system = (
        "You rewrite one section's body text for a set of study notes. Output ONLY "
        "the rewritten body -- no heading, no preamble, no meta-commentary about "
        "the rewrite itself."
    )

    try:
        new_body = call_gemini(prompt, system=system)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Regenerate request to Gemini failed: {exc}") from exc

    sections[index]["markdown_text"] = f"{heading_line}\n\n{new_body.strip()}"
    new_markdown = _save_edited_note(note_id, note["title"], sections)

    return {
        "note_id": note_id,
        "index": index,
        "markdown_text": sections[index]["markdown_text"],
        "notes_markdown": new_markdown,
    }


@app.patch("/api/notes/{note_id}/sections/{index}")
def edit_section(note_id: int, index: int, req: SectionEditRequest):
    """
    Pure manual edit -- no LLM call, no Gemini cost. The user's text replaces
    the section verbatim, heading included, so (unlike regenerate) this CAN
    change or remove the heading's [MM:SS–MM:SS] bracket. That's allowed on
    purpose: it just means the next extract_timed_sections() pass skips this
    section for the seek-chip timeline, the same graceful no-op untimed notes
    (pasted text/pdf/docx) already go through -- not an error condition.
    """
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")

    sections = export.split_into_sections(note["structured_markdown"])
    if index < 0 or index >= len(sections):
        raise HTTPException(status_code=404, detail="Section index out of range.")

    new_text = req.markdown_text.strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Section text can't be empty.")

    sections[index]["markdown_text"] = new_text
    new_markdown = _save_edited_note(note_id, note["title"], sections)

    return {
        "note_id": note_id,
        "index": index,
        "markdown_text": sections[index]["markdown_text"],
        "notes_markdown": new_markdown,
    }


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

@app.get("/api/library")
def library():
    """Everything the student has generated so far, newest first."""
    return {"items": db.list_library()}


@app.get("/api/notes/{note_id}")
def get_note(note_id: int):
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return note


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int):
    """Remove a note, its source, its question sets, its chat sessions, and its
    RAG chunks. The DB delete is what must succeed; a failed Chroma cleanup is
    logged rather than raised, so the note still disappears from the library."""
    if not db.delete_note(note_id):
        raise HTTPException(status_code=404, detail="Note not found.")

    try:
        rag.delete_note(note_id)
    except Exception as exc:
        print(f"[rag] failed to remove chunks for note {note_id}: {exc}")

    return {"deleted": note_id}


@app.post("/api/reindex")
def reindex():
    """Manually (re)build the RAG index from every note currently in the
    database. Useful for backfilling notes generated before RAG existed."""
    try:
        return rag.reindex_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Practice questions
# ---------------------------------------------------------------------------

@app.post("/api/generate-questions")
def generate_questions_endpoint(req: QuestionsRequest):
    """Generate practice questions from an already-generated note. Any source
    format works, because the note itself was built from one."""
    note = db.get_note(req.note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")

    learning_mode = normalize_learning_mode(req.learning_mode)
    count = questions_mod.normalize_count(req.count)
    from learning_modes import question_style   # or add to the top-level imports
    system = (
        f"{questions_mod.QUESTION_SYSTEM}\n\n"
        f"DIFFICULTY CALIBRATION: {question_style(learning_mode)}"
    )
    try:
        items = questions_mod.generate_questions(
            markdown=note["structured_markdown"],
            title=note["title"],
            count=count,
            call_llm=call_gemini,
            system_prompt=system,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Question generation failed: {exc}") from exc

    if not items:
        raise HTTPException(
            status_code=422,
            detail="The model didn't return usable questions. Try again.",
        )

    set_id = db.save_question_set(
        note_id=req.note_id,
        requested_count=count,
        learning_mode=learning_mode,
        questions=items,
    )

    return {
        "set_id": set_id,
        "note_id": req.note_id,
        "title": note["title"],
        "requested_count": count,
        "returned_count": len(items),
        "learning_mode": learning_mode,
        "questions": items,
    }


@app.get("/api/question-sets/{set_id}")
def get_question_set(set_id: int):
    data = db.get_question_set(set_id)
    if not data:
        raise HTTPException(status_code=404, detail="Question set not found.")
    return data


# ---------------------------------------------------------------------------
# Chat — general Q&A, or scoped to one note. Both RAG-backed.
# Conversations are sessions, so old threads stay reopenable.
# ---------------------------------------------------------------------------

CHAT_HISTORY_TURNS = 10   # how many past messages to feed back in as context
RAG_TOP_K_SCOPED = 5      # chunks retrieved when chat is scoped to one note
RAG_TOP_K_GENERAL = 6     # chunks retrieved when searching the whole library

GENERAL_CHAT_SYSTEM_PROMPT = (
    "You are a friendly, precise study assistant. Answer clearly and concisely, "
    "and show your reasoning for anything mathematical or technical. If you are "
    "not confident a term means what you think it means — especially acronyms, "
    "tool names, and library names that could refer to several things — say so "
    "and ask which one the student means, rather than guessing."
)

GENERAL_RAG_SYSTEM_PROMPT_TEMPLATE = (
    "You are a study assistant with access to the student's note library. Below are the "
    "passages most relevant to their question, pulled from one or more of their generated "
    "notes. Use them as your primary source of truth, and mention which note(s) a fact came "
    "from when it's useful. If the passages don't cover the question, say so, then answer "
    "from general knowledge.\n\n{context}"
)

NOTE_RAG_SYSTEM_PROMPT_TEMPLATE = (
    "You are a study assistant helping a student understand a specific set of notes titled "
    "\"{title}\". Below are the passages from those notes most relevant to their question. "
    "Answer using them as your primary source of truth. If the question can't be answered "
    "from these passages, say so, then answer from general knowledge.\n\n{context}"
)

# Used only as a fallback when the RAG index has nothing for this note yet
# (e.g. rag deps aren't installed, or indexing failed silently at generation time).
NOTE_WHOLE_NOTE_SYSTEM_PROMPT_TEMPLATE = (
    "You are a study assistant helping a student understand a specific set of notes. "
    "Answer using the notes below as your primary source of truth. If the question can't "
    "be answered from the notes, say so, then answer from general knowledge.\n\n"
    "--- NOTES: {title} ---\n{markdown}\n--- END NOTES ---"
)


def _format_context(hits: list) -> str:
    """Turn retrieved chunks into a labeled context block for the system prompt."""
    blocks = [f"[From \"{hit['title']}\"]\n{hit['chunk']}" for hit in hits]
    return "\n\n---\n\n".join(blocks)


def _dedup_sources(hits: list) -> list:
    """One entry per note_id, in the order it first appeared (best match first)."""
    seen = set()
    sources = []
    for hit in hits:
        if hit["note_id"] in seen:
            continue
        seen.add(hit["note_id"])
        sources.append({"note_id": hit["note_id"], "title": hit["title"]})
    return sources


@app.get("/api/chat/sessions")
def chat_sessions():
    """Every conversation, most-recently-active first (populates the dropdown)."""
    return {"sessions": db.list_chat_sessions()}


@app.post("/api/chat/sessions")
def new_chat_session(scope: str = "general"):
    """Explicitly create an empty session. The frontend doesn't need this —
    POST /api/chat creates one implicitly on the first message — but it's
    useful for testing."""
    return {"session_id": db.create_chat_session(scope)}


@app.get("/api/chat/history")
def chat_history(session_id: int):
    return {"messages": db.get_chat_history(session_id)}


@app.delete("/api/chat/sessions/{session_id}")
def delete_chat_session(session_id: int):
    if not db.delete_chat_session(session_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"deleted": session_id}


@app.post("/api/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message can't be empty.")

    # Resolve the session, creating one on the conversation's first message.
    # The session owns the scope, so req.scope only matters at creation time.
    session_id = req.session_id
    is_new = session_id is None
    if is_new:
        session_id = db.create_chat_session((req.scope or "general").strip())

    session = db.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    scope = session["scope"]

    sources: list = []

    if scope.startswith("note:"):
        try:
            note_id = int(scope.split(":", 1)[1])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chat scope.")

        note = db.get_note(note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found for this chat scope.")

        hits = []
        try:
            hits = rag.query(message, top_k=RAG_TOP_K_SCOPED, note_id=note_id)
        except Exception as exc:
            print(f"[rag] query failed, falling back to whole-note context: {exc}")

        if hits:
            system_prompt = NOTE_RAG_SYSTEM_PROMPT_TEMPLATE.format(
                title=note["title"], context=_format_context(hits)
            )
            sources = _dedup_sources(hits)
        else:
            system_prompt = NOTE_WHOLE_NOTE_SYSTEM_PROMPT_TEMPLATE.format(
                title=note["title"], markdown=note["structured_markdown"]
            )

    else:
        # General chat searches across the WHOLE library via RAG.
        hits = []
        try:
            hits = rag.query(message, top_k=RAG_TOP_K_GENERAL, note_id=None)
        except Exception as exc:
            print(f"[rag] query failed, falling back to plain general chat: {exc}")

        if hits:
            system_prompt = GENERAL_RAG_SYSTEM_PROMPT_TEMPLATE.format(
                context=_format_context(hits)
            )
            sources = _dedup_sources(hits)
        else:
            system_prompt = GENERAL_CHAT_SYSTEM_PROMPT

    learning_mode = normalize_learning_mode(req.learning_mode)
    system_prompt = apply_learning_mode(system_prompt, learning_mode)

    history = db.get_chat_history(session_id)[-CHAT_HISTORY_TURNS:]
    transcript = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history
    )
    prompt = (
        f"Conversation so far:\n{transcript}\n\nUser: {message}\nAssistant:"
        if transcript
        else message
    )

    try:
        reply = call_gemini(prompt, system=system_prompt, temperature=0.4)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat request to Gemini failed: {exc}") from exc

    db.save_chat_message(session_id, "user", message)
    db.save_chat_message(session_id, "assistant", reply)

    # Name the thread after its opening question so the dropdown is readable.
    if is_new:
        title = message[:60] + ("…" if len(message) > 60 else "")
        db.rename_chat_session(session_id, title)
    else:
        db.touch_chat_session(session_id)

    return {
        "reply": reply,
        "sources": sources,
        "session_id": session_id,
        "learning_mode": learning_mode,
    }


# ---------------------------------------------------------------------------
# Export
#
# NOTE: the questions route is declared FIRST on purpose. FastAPI matches in
# declaration order, and /api/export/{note_id}/docx would otherwise try to
# parse "questions" as an int for a URL like /api/export/questions/3/docx.
# ---------------------------------------------------------------------------

@app.get("/api/export/questions/{set_id}/{fmt}")
def export_questions(set_id: int, fmt: str, answers: bool = False):
    """Download a question set as .docx or .pdf. `answers=true` includes the
    answer key; the frontend passes whatever the Show/Hide toggle is set to."""
    data = db.get_question_set(set_id)
    if not data:
        raise HTTPException(status_code=404, detail="Question set not found.")

    if fmt not in ("docx", "pdf"):
        raise HTTPException(status_code=400, detail="Format must be docx or pdf.")

    markdown = questions_mod.to_markdown(
        title=data["title"],
        items=data["questions"],
        include_answers=answers,
    )

    suffix = "with-answers" if answers else "questions-only"
    filename = f"{_safe_filename(data['title'])}_{suffix}.{fmt}"
    doc_title = f"Practice questions — {data['title']}"

    if fmt == "docx":
        file_bytes = export.markdown_to_docx(markdown, doc_title)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        file_bytes = export.markdown_to_pdf(markdown, doc_title)
        media_type = "application/pdf"

    return Response(
        content=file_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/{note_id}/markdown")
def export_markdown(note_id: int):
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    filename = _safe_filename(note["title"]) + ".md"
    return Response(
        content=note["structured_markdown"],
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/{note_id}/docx")
def export_docx(note_id: int):
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    filename = _safe_filename(note["title"]) + ".docx"
    file_bytes = export.markdown_to_docx(note["structured_markdown"], note["title"])
    return Response(
        content=file_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/{note_id}/pdf")
def export_pdf(note_id: int):
    note = db.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    filename = _safe_filename(note["title"]) + ".pdf"
    file_bytes = export.markdown_to_pdf(note["structured_markdown"], note["title"])
    return Response(
        content=file_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _safe_filename(title: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return cleaned[:80] or "video-notes"


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Serve the frontend (static/index.html) at http://127.0.0.1:8000/
# Must be mounted LAST so it doesn't shadow the /api routes above.
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
