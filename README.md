# Notewell

**Live demo → [notewell-7p6e.onrender.com](https://notewell-7p6e.onrender.com)**

Turn a YouTube link, PDF, Word document, audio file, or pasted text into structured
study notes — then chat with them, generate practice questions, and export everything
to Markdown, DOCX, or PDF.

Built to run on a free stack: Gemini for the LLM, local Whisper for transcription,
local embeddings for retrieval, SQLite and ChromaDB for storage. No paid APIs beyond
a Gemini API key.

> **About the demo:** it's on Render's free tier, so the first request after a period
> of inactivity takes a minute or so while the server wakes and reloads its models.
> Nothing is persisted between restarts — anything you generate is yours for that
> session only.

> **About accuracy:** notes, chat replies, and practice questions are AI-generated and
> can contain confident mistakes, particularly around tools or terms with similar names.
> Check anything important against your original source.

---

## What it does

**Five input types.** YouTube links (uses captions where available, falls back to local
Whisper transcription where not), PDF, DOCX, audio uploads, and pasted text. Everything
funnels into the same chunk → summarize → structure pipeline.

**Timestamped, clickable notes for video sources.** Video notes carry real timecodes —
each section heading is labeled with the exact `[MM:SS–MM:SS]` range of the source it
covers, computed from the actual transcript timing, not guessed by the model. You can
also generate notes for just a slice of a video instead of the whole thing (`start_time`/
`end_time`). Every timestamped section renders as a clickable chip that opens the source
video at that exact moment, in a new tab.

**Live, streamed generation for video/audio sources.** Generating notes from a video
doesn't block on one long request. It runs as a background job: each section is written
in its final, ready-to-render form and streamed to the page the moment it's done, so you
can start reading early sections while later ones are still being written. A live,
self-calibrating time estimate updates as it goes, based on this run's actual measured
speed rather than a fixed guess.

**Three learning modes.** Beginner, Medium, and Expert produce genuinely different
documents, not the same notes at three lengths. The mode changes the *structure* of the
output, not just the wording: Beginner leads with a concrete example and includes a
glossary, Expert opens with a dense summary, drops the glossary as padding, and adds
sections on trade-offs and open questions. The mode also shapes chat replies and the
difficulty calibration of practice questions.

**Chat with your notes.** RAG-backed retrieval over a local vector store. Scope a
conversation to one note, or leave it general and it searches your whole library,
showing which notes it drew from. Conversations are saved as threads you can reopen.

**Practice questions.** Generate 10, 20, 50, or 100 from any saved note. Long sets are
produced in batches with previously-generated questions fed back in, so the model
doesn't repeat itself. Export to DOCX or PDF, with or without the answer key depending
on whether answers are shown on screen.

**Library.** Browse, reopen, and delete everything you've generated. Deleting a note
removes its question sets, its chat threads, and its vector-store chunks.

---

## Project structure

```
vtn/
├── backend/
│   ├── main.py              FastAPI app — all routes, background job pipeline
│   ├── db.py                SQLite persistence + schema migrations
│   ├── rag.py               Chunking, embeddings, ChromaDB retrieval
│   ├── questions.py         Practice-question generation + Markdown rendering
│   ├── learning_modes.py    Beginner/Medium/Expert prompts and note structures
│   ├── extraction.py        PDF and DOCX text extraction
│   ├── transcription.py     yt-dlp + faster-whisper, incremental progress reporting
│   ├── export.py            Markdown → DOCX / PDF, timestamped-section extraction
│   ├── timeutils.py         Timestamp parsing/formatting, adaptive marker density
│   ├── requirements.txt
│   ├── .env.example
│   ├── video_to_notes.db    created on first run (gitignored)
│   └── chroma_db/           created on first run (gitignored)
├── static/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
└── README.md
```

---

## Running it locally

### Prerequisites

- Python 3.10+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **`ffmpeg` on your system PATH.** Not installed via pip — it's a system binary, needed
  for trimming audio when you request a custom time range. `winget install ffmpeg`
  (Windows), `brew install ffmpeg` (macOS), or `apt-get install ffmpeg` (Linux). After
  installing, open a **new** terminal before running the server — an already-open shell
  keeps the old `PATH` cached and won't see the newly installed binary.
- ~1 GB free disk space for model downloads (one-time, cached afterwards)

### Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Copy the environment template and add your key:

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

The Whisper settings in `.env.example` (`WHISPER_MODEL_SIZE`, `WHISPER_DEVICE`,
`WHISPER_COMPUTE_TYPE`) are optional — the defaults (small / cpu / int8) work as-is.

### Run

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000**. The database and vector store are created automatically
on first run.

Two models download from Hugging Face on first use, not at install time:
`all-MiniLM-L6-v2` for embeddings (~90 MB) and a Whisper model the first time you
process something with no captions.

---

## How it works

### The notes pipeline

Any source is reduced to plain text, split into chunks, and assembled into structured
Markdown. Two variants:

- **Untimed sources** (pasted text, PDF, DOCX): word-count chunking, summarized chunk by
  chunk (the map step), then assembled into one structured document in a single reduce
  call. The map step is deliberately **not** mode-aware — simplifying there loses detail
  the reduce step can never recover, an early version produced Expert notes that were
  vaguer than Medium because specifics had already been dropped.
- **Timed sources** (video, audio): chunked by *time* instead of word count, with each
  chunk written directly into its final, ready-to-render section — see below.

### Timestamps and custom ranges

Both transcript sources (YouTube captions and Whisper) carry per-segment timing
natively; it's kept rather than flattened into one string. From there:

- **Custom range** — pass `start_time`/`end_time` (accepts `"12:30"`, `"1:02:00"`, or
  plain seconds) to `/api/generate-notes` to generate notes for just a slice of a video.
  Captions are filtered in place; for Whisper, the downloaded audio is trimmed with
  `ffmpeg` before transcription, so only the requested slice is actually processed.
- **Adaptive marker density** — rather than a fixed chunk size, the interval is picked
  so a note ends up with roughly 8 section headings regardless of video length: a
  12-minute clip gets ~5-minute sections, a 3-hour lecture gets ~30-minute sections.
- **Section headings carry real timecodes** in the form `[MM:SS–MM:SS] Title`, extracted
  after generation into a `note_sections` table (not regex-parsed by the frontend).
- **Clickable timestamps** — the frontend renders each section as a chip linking to
  `youtube.com/watch?v={id}&t={seconds}s`, opening in a new tab already seeked to that
  moment. Deliberately not an embedded player: the YouTube iframe API can be silently
  blocked by ad blockers or restrictive networks, which turned out to be more fragile
  than it was worth for what a plain link achieves just as well.

### Async generation & live progress

Video/audio generation runs as a background job instead of one blocking request:

- `POST /api/generate-notes` returns `{"job_id": ...}` immediately; the pipeline runs on
  a background thread.
- `GET /api/jobs/{job_id}` is polled by the frontend every ~2 seconds, returning current
  stage (`transcribing` / `writing_sections` / `finalizing` / `done`), a live time
  estimate, and every section written so far.
- Each chunk is written directly in **final, ready-to-render form** — not an
  intermediate summary to be rewritten later — so the frontend can render real content
  as it streams in, not just a percentage. The whole-document parts that need full
  context (the opening TL;DR/summary, and the closing glossary/trade-offs/check-yourself
  questions) are written in one pass at the end, once every section exists to reference.
- The time estimate is **not** a hardcoded guess — Whisper's speed varies too much by
  CPU for that to be meaningful. It's computed from this run's own observed throughput
  (`elapsed ÷ work done so far`), so it self-calibrates within the first few data points
  rather than showing a number nobody could back up.
- Untimed sources (pasted text, PDF, DOCX, direct audio upload) stay simple synchronous
  requests — they don't have the long-wait problem this solves.

### Learning modes

Each mode supplies prose-style instructions and a Markdown skeleton. For untimed
sources, one skeleton covers the whole document. For timed sources, the skeleton is
split three ways — one for writing a single section, one for the opening block(s), one
for the closing block(s) — so a chunk's section can be written and streamed the moment
it's ready, with only the whole-document parts (TL;DR, glossary, trade-offs) waiting
until every section exists.

The skeleton is what matters more than the tone instructions. An earlier version varied
only the system prompt and the three modes came out nearly identical, because a
hardcoded output structure in the user prompt overrides tone guidance in the system
prompt. Explicit shape instructions belong in the user prompt.

### RAG

`rag.py` splits each note into ~220-word overlapping chunks, breaking on headings first
so a chunk doesn't straddle unrelated sections. Chunks are embedded locally with
`sentence-transformers` and stored in ChromaDB, filtered by `note_id` metadata for
note-scoped chat.

Reindexing is always delete-then-upsert (`reindex_note_clean`), not upsert-only —
upserting alone can leave stale, orphaned chunks behind if a note's chunk count ever
shrinks. If the embedding dependencies are missing or a note hasn't been indexed, chat
falls back to passing the whole note as context. It degrades rather than breaking.

`POST /api/reindex` rebuilds the index from every note in the database. It also runs
once in a background thread at startup, so notes generated before RAG existed get
picked up automatically.

### Question generation

100 questions in one LLM call produces repetition and drift, so sets are generated in
batches of 20 with already-written questions passed back in as things to avoid. Output
is JSON, parsed defensively — models wrap it in code fences despite instructions not to.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/generate-notes` | From a YouTube URL. Accepts optional `start_time`/`end_time`. Starts a background job, returns `{"job_id": ...}` immediately |
| `GET` | `/api/jobs/{job_id}` | Poll a video-generation job — stage, live time estimate, streamed sections, final `note_id` once done |
| `GET` | `/api/notes/{id}/sections` | Timestamped section list for a note (empty for untimed sources) |
| `POST` | `/api/generate-notes/text` | From pasted text |
| `POST` | `/api/generate-notes/pdf` | From an uploaded PDF |
| `POST` | `/api/generate-notes/docx` | From an uploaded Word document |
| `POST` | `/api/generate-notes/audio` | From an uploaded audio file |
| `GET` | `/api/library` | All notes, newest first |
| `GET` | `/api/notes/{id}` | One note |
| `DELETE` | `/api/notes/{id}` | Delete a note and everything derived from it |
| `POST` | `/api/generate-questions` | Practice questions from a note |
| `GET` | `/api/question-sets/{id}` | Retrieve a saved set |
| `GET` | `/api/chat/sessions` | All conversations |
| `GET` | `/api/chat/history?session_id=` | Messages in one conversation |
| `POST` | `/api/chat` | Send a message (creates a session if none given) |
| `DELETE` | `/api/chat/sessions/{id}` | Delete a conversation |
| `POST` | `/api/reindex` | Rebuild the vector index |
| `GET` | `/api/export/{id}/{md\|docx\|pdf}` | Export a note |
| `GET` | `/api/export/questions/{id}/{docx\|pdf}?answers=` | Export a question set |

---

## Deploying

The live demo runs on Render's free tier:

- **Root directory:** `backend`
- **Build:** `apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt`
  (the `ffmpeg` install is required for custom time-range trimming — Render's default
  image doesn't include it)
- **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment:** `GEMINI_API_KEY`

`requirements.txt` pins the CPU-only PyTorch index. Without that line, Linux pulls the
CUDA build — about 2.5 GB of NVIDIA libraries this app never touches, which won't fit in
a free-tier container.

Free tier has no persistent disk, so the SQLite database and ChromaDB store reset on
every restart. Fine for a demo where visitors generate their own notes; a paid plan with
a mounted disk, or an external Postgres, would be needed to keep anything.

---

## Troubleshooting

**"GEMINI_API_KEY is not set"** — `.env` is missing, or it isn't inside `backend/`.

**Gemini model/API errors** — confirm `GEMINI_API_KEY` is valid and that
`GEMINI_MODEL` names an available Gemini model. The default is `gemini-2.5-flash`.

**`FileNotFoundError: [WinError 2]` / "ffmpeg not recognized"** — `ffmpeg` isn't on
`PATH`. Install it (see Prerequisites), then open a genuinely new terminal — and restart
the server from that new terminal — before trying again. An already-running shell or
server process keeps the old `PATH` cached even after installation.

**"Failed building wheel for av" / "Microsoft Visual C++ 14.0 required"** — an old
`faster-whisper` pin resolving to an `av` version with no Windows wheel. Run
`pip install --upgrade pip`, then reinstall from this `requirements.txt`.

**Whisper is slow** — expected on CPU, especially the first run while the model
downloads. Drop `WHISPER_MODEL_SIZE` to `base` or `tiny` in `.env` to trade accuracy for
speed. This is also why generation for video runs as a background job with live progress
rather than blocking — see "Async generation & live progress" above.

**"Could not download audio for this video" / HTTP 403 from yt-dlp** — often a stale
`yt-dlp` version; YouTube changes its player logic specifically to break scrapers, and
`yt-dlp` patches around it continuously. Try `pip install -U yt-dlp` first. If that
doesn't resolve it, the video itself may be age-restricted, region-locked, or have
owner-disabled downloads.

**Exports look wrong** — the exporters parse the Markdown the model produces (`##`
headings, `-` bullets, `**bold**`). Heavily hand-edited notes may not convert cleanly.

**CORS or connection errors** — open `http://127.0.0.1:8000`, not `static/index.html`
from disk.

**Chat cites a note you deleted** — the Chroma delete failed silently. Check the console
for a `[rag]` line and run `POST /api/reindex`.

---

## Known limitations

- **Scanned PDFs don't work.** Text extraction only, no OCR.
- **The model invents connections between similarly-named things.** Observed with
  uv/libuv, uv/Uvicorn, and LangChain/LangGraph. Prompt constraints reduce it; nothing
  eliminates it at this model size.
- **No embedded video player.** Timestamp chips open the source video in a new tab
  rather than playing inline, to avoid the YouTube iframe API's dependency on
  third-party requests that ad blockers and some networks silently block.
- **Custom time ranges still download the full audio** before trimming, for videos that
  fall back to Whisper — only transcription is limited to the requested slice, not the
  download. A range-aware partial download is a possible future optimization.
- **No authentication.** Single-user by design; the learning-mode preference lives in
  `localStorage`.

## Possible next steps

- **Section-level edit/regenerate** — revise or manually edit one section of a note
  without touching the rest or regenerating from scratch.
- **Batch playlist import** — submit a whole YouTube playlist, processed in the
  background with per-video progress and per-item failure isolation.
- **Partial audio download for custom ranges** — download only the requested slice's
  bytes instead of the full audio before trimming.
- Streaming chat replies
- Interactive quiz mode — answer questions in-app rather than reading a list
- Flashcards generated from the Key Terms section
- Hosted transcription instead of local Whisper, which would cut the container size
  dramatically and make audio more viable on free hosting
