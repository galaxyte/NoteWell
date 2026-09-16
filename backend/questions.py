"""Question generation — turns a note's markdown into practice questions.

Large counts (50/100) are generated in batches, feeding already-generated
questions back into each prompt so the model doesn't repeat itself.
"""

import json
import re
import textwrap
from typing import Callable, List

ALLOWED_COUNTS = (10, 20, 50, 100)
BATCH_SIZE = 20

QUESTION_SYSTEM = (
    "You are an experienced exam-setter. You write clear, unambiguous practice "
    "questions that test real understanding, not trivia recall. You reply with "
    "raw JSON only — no markdown fences, no commentary."
)


def normalize_count(count) -> int:
    try:
        count = int(count)
    except (TypeError, ValueError):
        return 10
    return count if count in ALLOWED_COUNTS else 10


def _parse_json_array(raw: str) -> List[dict]:
    """Models sometimes wrap JSON in ```json fences despite instructions."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("question")]


def _batch_prompt(markdown: str, title: str, n: int, already: List[dict]) -> str:
    avoid = ""
    if already:
        recent = "\n".join(f"- {q['question']}" for q in already[-40:])
        avoid = f"\n\nDo NOT repeat or rephrase any of these already-written questions:\n{recent}"

    return textwrap.dedent(f"""
        Study material titled "{title}":

        ---
        {markdown}
        ---

        Write exactly {n} practice questions based on this material.

        Mix the types:short-answer,medium level questions and question asked for last 10 years on the same topic.
        Vary difficulty across easy / medium / hard.

        Reply with a JSON array only. Each element must be an object:
        {{
          "question": "the question text",
          "type": "mcq" or "short",
          "difficulty": "easy" | "medium" | "hard",
          "answer": "the correct answer with explanation",

        }}
    """).strip() + avoid


def generate_questions(
    *,
    markdown: str,
    title: str,
    count: int,
    call_llm: Callable[[str, str], str],
    system_prompt: str = QUESTION_SYSTEM,
) -> List[dict]:
    """call_llm(prompt, system) -> str. Injected so this module stays
    independent of main.py's Gemini client (avoids a circular import)."""
    count = normalize_count(count)
    collected: List[dict] = []
    remaining = count

    while remaining > 0 and len(collected) < count:
        n = min(BATCH_SIZE, remaining)
        raw = call_llm(_batch_prompt(markdown, title, n, collected), system_prompt)
        batch = _parse_json_array(raw)
        if not batch:
            break  # model returned unparseable output; stop rather than loop forever
        collected.extend(batch)
        remaining -= len(batch)

    return collected[:count]

def to_markdown(*, title: str, items: List[dict], include_answers: bool) -> str:
    """Render a question set as Markdown, ready for the docx/pdf exporters."""
    lines = [f"# Practice questions — {title}", ""]
    if not include_answers:
        lines += ["*Answers not included.*", ""]

    for i, q in enumerate(items, start=1):
        diff = f" _({q['difficulty']})_" if q.get("difficulty") else ""
        lines.append(f"**{i}.** {q.get('question', '')}{diff}")
        lines.append("")

        for opt in q.get("options") or []:
            lines.append(f"- {opt}")
        if q.get("options"):
            lines.append("")

        if include_answers:
            lines.append(f"**Answer:** {q.get('answer', '—')}")
            if q.get("explanation"):
                lines.append("")
                lines.append(f"{q['explanation']}")
            lines.append("")

        lines.append("")

    return "\n".join(lines)
