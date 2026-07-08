import logging
import json
import io
import re
from datetime import date, timedelta, timezone, datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from openai import OpenAI
from pypdf import PdfReader

from app.config import settings
from app.database import get_db
from app.models import Course, Syllabus, Lesson, Annotation, Feedback, LearningEvent
from app.schemas import (
    CreateCourseRequest, CourseResponse, CourseDetailResponse, CreateSourceCourseResponse,
    SyllabusResponse, SyllabusUpdateRequest,
    LessonListItem, LessonResponse,
    CreateAnnotationRequest, AnnotationResponse, AddAnnotationMessageRequest, SaveInterruptedRequest,
    CreateFeedbackRequest, FeedbackResponse,
    CourseStatsResponse, GlobalStatsResponse,
    CalendarResponse, CalendarDay, CalendarCourseActivity,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["courses"])

LEARNING_DEPTH_PROFILES = {
    "simple": {
        "label": "Simple",
        "modules": "2–3 modules",
        "items": "8–10 mastery items",
        "focus": "Retain only the core trunk of the topic, the minimum necessary concepts, and high-frequency applications; avoid historical context, complex proofs, fringe debates, and advanced extensions.",
    },
    "standard": {
        "label": "Standard",
        "modules": "3–4 modules",
        "items": "10–12 mastery items",
        "focus": "Cover core concepts, key reasoning, typical applications, and common misconceptions; balance completeness with cognitive load.",
    },
    "deep": {
        "label": "Deep",
        "modules": "4–5 modules",
        "items": "12–15 mastery items",
        "focus": "Build from first principles; include underlying mechanisms, boundary conditions, counterexamples, cross-context transfer, and critical judgment.",
    },
}


def _learning_depth_profile(learning_depth: str) -> dict[str, str]:
    if learning_depth not in LEARNING_DEPTH_PROFILES:
        raise HTTPException(status_code=400, detail="Learning depth must be simple, standard, or deep")
    return LEARNING_DEPTH_PROFILES[learning_depth]


def _build_syllabus_prompt(learning_depth: str) -> str:
    profile = _learning_depth_profile(learning_depth)
    depth_section = f"""- Learning depth: {profile['label']}
- Number of modules: {profile['modules']}
- Number of mastery items: {profile['items']}
- Expansion strategy: {profile['focus']}"""
    return SYLLABUS_PROMPT.format(learning_depth_section=depth_section)


# ---------------------------------------------------------------------------
# AI System Prompts
# ---------------------------------------------------------------------------

SYLLABUS_PROMPT = """You are an expert course syllabus designer. Based on the topic name given by the user, generate a structured course syllabus.

## Learning Depth (must be followed)
{learning_depth_section}

## Output Format (follow strictly — no extra commentary)

```markdown
# [Topic Name] · Course Syllabus

> This syllabus defines all the competencies you will have mastered upon completing this topic.
> Learning depth: [Simple / Standard / Deep]
> The number of lessons varies by person, but the learning content is never compromised.

## Core Mastery Items

After completing this topic, you will be able to:

### [Module One Name]
- [ ] [Specific competency description, phrased as "be able to…", verifiable]
- [ ] [Specific competency description]

### [Module Two Name]
- [ ] [Specific competency description]
- [ ] [Specific competency description]

## Out of Scope

- [Clearly list which related topics are not covered in this course]

## Learning Progress

| Lesson | Mastery Items Covered | Date Generated |
|--------|----------------------|----------------|
```

## Rules
1. All mastery items must be **verifiable behaviors** (can explain, derive, apply, or judge) — "understand X" or "be familiar with Y" are forbidden
2. Module count must follow the learning depth specified above, grouped by the internal logic of the knowledge
3. Total item count must follow the learning depth specified above, and all items must be substantively distinct
4. "Out of Scope" must be filled in
5. Output only markdown content — no prefix or suffix explanations
"""

FIRST_LESSON_PROMPT = """You are a one-on-one Socratic tutor grounded in Bloom's 2-Sigma theory.

Based on the course syllabus below, generate the first lesson (01).

## Course Syllabus
{syllabus}

## Output Format (follow strictly)

```markdown
# [Chapter Title]

> Prerequisites: [list the prerequisites needed to read this lesson]
> Difficulty: [Beginner / Intermediate / Advanced]
> Estimated reading time: [X minutes]

## Content

[Clear, substantive, example-rich knowledge exposition]
[Key concepts in **bold**]
[Important definitions or formulas in blockquotes]

## Reflection Questions

[2–3 questions that guide the reader into deeper thinking — no answers provided]

## Your Feedback

> Write your questions, insights, things you didn't understand, or directions you'd like the next lesson to explore.
```

## Rules
1. Content must have substantive knowledge value — avoid superficiality
2. Bold key concepts; put important definitions in blockquotes
3. Reflection questions must be thought-provoking, not simple recall
4. Output only markdown content
5. **Analogy first**: every abstract concept gets at least one everyday analogy
6. **Why before what**: explain why this topic matters before explaining the content
7. **Cognitive load control**: the first lesson introduces only 2–3 core concepts — don't spread too wide
8. **Consistent depth**: the first lesson's depth must match the learning depth in the syllabus; simple = trunk only, standard = complete coverage, deep = underlying mechanisms and edge cases
"""

SOURCE_LESSON_PROMPT = """You are a one-on-one Socratic tutor grounded in Bloom's 2-Sigma theory.

The student has just finished reading a user-uploaded PDF/TXT/Markdown source document. Based on the full text, the student's highlighted questions, and your immediate answers to those questions, generate the next lesson.

## Course Syllabus
{syllabus}

## Source Document File
{source_filename}

## Full Source Document
{source_content}

## Student Highlight Q&A Record
{source_annotations}

## Current Lesson Number: {lesson_number}

## Output Format (follow strictly)

```markdown
# [Chapter Title]

> Prerequisites: [list the prerequisites needed to read this lesson]
> Difficulty: [Beginner / Intermediate / Advanced]
> Estimated reading time: [X minutes]

---

## Highlight Review

> This section synthesizes the student's highlighted questions from the source document to surface genuine comprehension gaps.

[First identify common misconceptions behind the student's questions, then correct key misunderstandings. Do not mechanically repeat each Q&A pair.]

---

## Content

[Clear, substantive, example-rich knowledge exposition. Extract the key threads from the source material rather than broadly discussing the topic.]

## Reflection Questions

[2–3 questions that guide the reader into deeper thinking — no answers provided]

## Your Feedback

> Write your questions, insights, things you didn't understand, or directions you'd like the next lesson to explore.

<!-- mastery: be able to...; be able to... -->
```

## Rules
1. Treat the source material as the primary textbook — do not improvise freely from the topic name alone
2. Incorporate the highlight Q&A record; prioritize closing comprehension gaps the student has already revealed
3. The main content introduces only 1–2 core concepts at a time to avoid overload
4. The hidden mastery comment must list the exact mastery item text from the syllabus covered in this lesson, matching the syllabus checkbox text exactly
5. Output only markdown content
"""

NEXT_LESSON_PROMPT = """You are a one-on-one Socratic tutor grounded in Bloom's 2-Sigma theory.

Based on the student's feedback and annotations, generate the next lesson.

## Course Syllabus
{syllabus}

## Completed Lessons
{previous_lessons}

## Student Feedback on the Previous Lesson
{feedback}

## Student Annotations on the Previous Lesson (text the student highlighted while reading)
{annotations}

## Actual Reflection Questions from the Previous Lesson (the review section must map to these exactly)
{last_questions}

## Current Lesson Number: {lesson_number}

## Output Format (follow strictly)

```markdown
# [Chapter Title]

> Prerequisites: [list the prerequisites needed to read this lesson]
> Difficulty: [Beginner / Intermediate / Advanced]
> Estimated reading time: [X minutes]

---

## Previous Lesson Reflection Review

> 📝 This section evaluates your answers to the previous lesson's reflection questions and provides correct answers.

### Your Answer Evaluation

[Evaluate each answer in turn: ✅ correct / ❌ incorrect / ⚠️ partially correct, with a brief explanation]
[If the student did not answer, note "Not answered" and give the correct answer directly]

### Correct Answers

[Review each question strictly following the "Actual Reflection Questions from the Previous Lesson": review exactly as many questions as appeared in the previous lesson, in the same order and wording, with none omitted, added, or reworded into a generalized theme. Even if the student did not answer, provide the correct answer for each question. Format:]

**Question 1:** [brief description of the question (preserve the key point of the original — do not reword into a generalized theme)]
> [Complete correct answer and necessary analysis]

**Question 2:** [brief description]
> [Complete correct answer and necessary analysis]

(…continue until all questions from the "Actual Reflection Questions from the Previous Lesson" are covered)

---

## Annotation Responses

> 💬 This section addresses all comprehension questions the student marked in the previous lesson.

[If no annotations, write "No annotations in the previous lesson. Proceeding to new content."]

---

## Content

[Clear, substantive, example-rich knowledge exposition]

## Reflection Questions

[2–3 questions that guide the reader into deeper thinking — no answers provided]

## Your Feedback

> Write your questions, insights, things you didn't understand, or directions you'd like the next lesson to explore.
```

## Rules

**[HIGHEST PRIORITY] The "Previous Lesson Reflection Review" must map strictly to the "Actual Reflection Questions from the Previous Lesson": review exactly as many questions as appeared, in the same order and with the same wording, with none omitted, added, or reworded into a generalized theme. Even if the student did not answer, provide the correct answer for each question.**

1. The order must be strictly: Reflection Review → Annotation Responses → New Content
2. Adjust content depth and direction based on student feedback and annotations
3. Each lesson should cover at least one mastery item from the syllabus
4. Output only markdown content

## Teaching Quality Requirements
5. **Analogy first**: every abstract concept gets at least one everyday analogy or concrete scenario to build intuition
6. **Adaptive difficulty**: if the student's feedback shows comprehension difficulty (many annotations, many questions in feedback), lower the difficulty and add more foundational groundwork; if the student shows ease, raise the challenge appropriately
7. **Cognitive load control**: the main content introduces only 1–2 new concepts at a time — no information overload
8. **Why before what**: explain why the concept is needed (motivation/problem) before explaining the concept itself
9. **Question tiers**: at least one question is application-level (applying the concept to a new scenario) and at least one is analysis-level (comparing, judging, reasoning)
10. **Mastery item marker**: append a hidden comment block at the very end of the lesson (after the feedback section) listing which mastery items from the syllabus this lesson covers (text must match the syllabus exactly), in the format:
    <!-- mastery: be able to explain core concept A; be able to apply concept A to solve simple problems -->
"""

ANNOTATION_ANSWER_PROMPT = """You are a one-on-one learning tutor. The student is reading material and has asked an immediate question about something they highlighted.

## Response Requirements
1. Answer the question directly — first clarify what this passage means, then add necessary background
2. Do not digress into the broader course unless it is necessary to understand this specific highlighted passage
3. If the student's question contains a misconception, point it out gently but clearly
4. Keep your response to 2–5 paragraphs
5. Do not output markdown headings
"""

EVAL_LESSON_PROMPT = """You are a one-on-one Socratic tutor grounded in Bloom's 2-Sigma theory.

All mastery items in the syllabus have been fully covered. Now generate the evaluation lesson — it only needs to address the final lesson's reflection questions and annotation questions, and contains no new content.

## Course Syllabus
{syllabus}

## Previous Lesson
{last_lesson}

## Student Feedback on the Previous Lesson
{feedback}

## Student Annotations on the Previous Lesson
{annotations}

## Actual Reflection Questions from the Previous Lesson (the review must map to these strictly, question by question)
{last_questions}

## Output Format (follow strictly — the first line must be <!-- eval-article -->)

```markdown
<!-- eval-article -->

# [Topic Name] · Final Evaluation

> This is the course evaluation lesson — it contains no new content.
> Purpose: to address the final lesson's reflection questions and annotations, confirming you have fully mastered the material.

---

## Previous Lesson Reflection Review

> 📝 This section evaluates your answers to the previous lesson's reflection questions and provides correct answers.

### Your Answer Evaluation

[Evaluate each answer in turn: ✅ / ❌ / ⚠️ with a brief explanation]

### Correct Answers

[Review each question strictly following the "Actual Reflection Questions from the Previous Lesson": review exactly as many questions as appeared, in the same order and with the same wording, with none omitted, added, or reworded into a generalized theme. Even if the student did not answer, provide the correct answer for each question. Format:]

**Question 1:** [brief description]
> [Complete answer and analysis]

(…continue until all questions from the "Actual Reflection Questions from the Previous Lesson" are covered)

---

## Annotation Responses

> 💬 This section addresses all comprehension questions the student marked in the previous lesson.

[If no annotations, write "No annotations in the previous lesson."]

---

## Your Feedback

> Write your final thoughts about this course, any remaining questions, or directions you'd like to explore further.
> When you finish reading this lesson, tell me "I've finished reading," and the system will automatically generate a complete summary for you.
```
"""

SUMMARY_PROMPT = """You are a course summary expert. Based on the complete learning journey of the course, generate a structured learning summary.

## Course Syllabus
{syllabus}

## All Lesson Content
{all_lessons}

## Output Format

```markdown
# [Topic Name] · Learning Summary

## Knowledge Map

[Core concepts and their relationships, expressed as a hierarchical list]

## Syllabus Review

[Review the achievement status of each mastery item]

## Key Insights

[The most important discoveries and understandings from the learning process]

## Further Directions

[Directions worth continuing to explore]
```

Output only markdown content.
"""


# ---------------------------------------------------------------------------
# LLM Helpers
# ---------------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)


_FENCE_RE = re.compile(r'^```(?:markdown|md)?\s*\n?', re.IGNORECASE)
_FENCE_END_RE = re.compile(r'\n?```\s*$')


def _strip_markdown_fences(text: str) -> str:
    """Strip wrapping ```markdown ... ``` fences that LLMs often add."""
    text = text.strip()
    if _FENCE_RE.match(text):
        text = _FENCE_RE.sub('', text, count=1)
        text = _FENCE_END_RE.sub('', text)
    return text.strip()


def _extract_thought_questions(content: str) -> str:
    """从课文 markdown 中抽取 `## 思考题` 区块的全文（截止到下一个二级标题或文末）。

    思考题通常位于课文末尾，单独抽取后作为独立字段传给模型，避免被正文截断丢失，
    确保下一篇能逐题复盘真实思考题（issue #3）。
    """
    if not content:
        return ""
    m = re.search(r'^##\s*Reflection Questions\s*$\n(.*?)(?=^##\s|\Z)', content, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _stream_llm(system_prompt: str, user_message: str):
    """Generator that yields SSE chunks from LLM streaming."""
    client = get_openai_client()
    stream = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )
    full_response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content, False
    yield full_response, True  # final yield with complete text


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Non-streaming LLM call, returns full response."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def _call_llm_messages(system_prompt: str, history: list[dict]) -> str:
    """Non-streaming LLM call with a multi-turn conversation history."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, *history],
    )
    return response.choices[0].message.content


def _stream_llm_messages(system_prompt: str, history: list[dict]):
    """Streaming multi-turn LLM call. Yields (chunk, False) then (full_text, True)."""
    client = get_openai_client()
    stream = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, *history],
        stream=True,
    )
    full_response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content, False
    yield full_response, True


def _extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


async def _extract_upload_text(file: UploadFile) -> tuple[str, str]:
    """Return (filename, extracted text) for a PDF, TXT, or Markdown upload."""
    filename = file.filename or "uploaded-source"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    raw = await file.read()

    if suffix in ("txt", "md", "markdown"):
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="Text file encoding not recognized; convert to UTF-8 and retry")
    elif suffix == "pdf":
        try:
            text = _extract_pdf_text(raw)
        except Exception:
            raise HTTPException(status_code=400, detail="PDF parsing failed; confirm the file is not encrypted and contains extractable text")
    else:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, or MD files are supported")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No readable text extracted from the file")
    return filename, text


_CODE_LANGS = {
    "py": "python", "js": "javascript", "jsx": "jsx", "ts": "typescript", "tsx": "tsx",
    "java": "java", "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "hpp": "cpp",
    "cs": "csharp", "go": "go", "rs": "rust", "rb": "ruby", "php": "php", "swift": "swift",
    "kt": "kotlin", "scala": "scala", "sh": "bash", "bash": "bash", "zsh": "bash",
    "sql": "sql", "html": "html", "css": "css", "scss": "scss", "vue": "vue",
    "json": "json", "yaml": "yaml", "yml": "yaml", "toml": "toml", "xml": "xml",
    "r": "r", "lua": "lua", "dart": "dart", "ex": "elixir", "clj": "clojure",
}


def _source_lesson_content(filename: str, source_text: str) -> str:
    return f"""# Source Material: {filename}

> Prerequisites: None
> Difficulty: determined by source material
> Estimated reading time: determined by source length

## Source Text

{source_text}

## Your Feedback

> While reading, select text to ask questions. When done, click "I've finished reading" and the system will generate the next lesson based on the full text and your highlight Q&A.
"""


def _load_messages(annotation: Annotation) -> list[dict]:
    """Parse stored message thread; fall back to comment/answer for old rows."""
    if annotation.messages:
        try:
            data = json.loads(annotation.messages)
            if isinstance(data, list) and data:
                return data
        except (ValueError, TypeError):
            pass
    history = [{"role": "user", "content": annotation.comment}]
    if annotation.answer:
        history.append({"role": "assistant", "content": annotation.answer})
    return history


def _annotation_to_response(annotation: Annotation) -> AnnotationResponse:
    return AnnotationResponse(
        id=annotation.id,
        lesson_id=annotation.lesson_id,
        position_start=annotation.position_start,
        position_end=annotation.position_end,
        original_text=annotation.original_text,
        comment=annotation.comment,
        answer=annotation.answer or "",
        messages=_load_messages(annotation),
        anchor_top=annotation.anchor_top or 0,
        pdf_position=annotation.pdf_position,
        created_at=annotation.created_at,
    )


def _format_annotations(annotations: list[Annotation]) -> str:
    if not annotations:
        return "No highlight Q&A on record."
    blocks = []
    for item in annotations:
        history = _load_messages(item)
        turns = "\n".join(
            f"  {'Question' if m['role'] == 'user' else 'Answer'}: {m['content']}" for m in history
        )
        blocks.append(f"- Highlighted text: \"{item.original_text}\"\n{turns}")
    return "\n".join(blocks)


def _annotation_system_prompt(course: Course, lesson: Lesson, selected_text: str) -> str:
    """Build the system prompt for a highlight Q&A turn: tutor instructions + full lesson + selection."""
    context = lesson.content
    if course.mode == "source" and lesson.is_source and not course.is_project and course.source_content:
        context = course.source_content
    if not context:
        context = selected_text
    return f"""{ANNOTATION_ANSWER_PROMPT}

## Current Learning Material
{context}

## Student Highlighted Text
{selected_text}
"""


def _count_mastery_items(syllabus_content: str) -> tuple[int, int]:
    """Count (checked, total) mastery checkbox items in syllabus."""
    checked = 0
    total = 0
    for line in syllabus_content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            total += 1
        elif stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            checked += 1
            total += 1
    return checked, total


def _check_all_mastery_items_done(syllabus_content: str) -> bool:
    """Check if all checkbox items in syllabus are checked."""
    checked, total = _count_mastery_items(syllabus_content)
    return total > 0 and checked == total


def _mastery_progress(syllabus_content: str) -> float:
    """Return mastery progress as 0.0 to 1.0."""
    checked, total = _count_mastery_items(syllabus_content)
    return checked / total if total > 0 else 0.0


def _record_event(db: Session, course_id: int, event_type: str, lesson_number: int | None = None):
    """Record a learning event."""
    db.add(LearningEvent(course_id=course_id, lesson_number=lesson_number, event_type=event_type))
    db.flush()


_MASTERY_COMMENT_RE = re.compile(r'<!--\s*mastery:\s*(.+?)\s*-->', re.IGNORECASE)


def _auto_check_mastery(syllabus_content: str, lesson_content: str) -> str:
    """Parse mastery items from lesson content and check them in syllabus."""
    match = _MASTERY_COMMENT_RE.search(lesson_content)
    if not match:
        return syllabus_content

    items = [item.strip() for item in match.group(1).split(";") if item.strip()]
    if not items:
        return syllabus_content

    updated = syllabus_content
    for item in items:
        # Replace "- [ ] ...item..." with "- [x] ...item..."
        unchecked = f"- [ ] {item}"
        checked = f"- [x] {item}"
        if unchecked in updated:
            updated = updated.replace(unchecked, checked, 1)

    return updated


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@router.post("/courses", response_model=CourseDetailResponse)
def create_course(req: CreateCourseRequest, db: Session = Depends(get_db)):
    """Create course + AI generates syllabus + first lesson (blocking)."""
    depth_profile = _learning_depth_profile(req.learning_depth)
    course = Course(name=req.name, mode="topic", status="learning", learning_depth=req.learning_depth)
    db.add(course)
    db.flush()

    try:
        ref_section = ""
        if req.reference.strip():
            ref_section = f"\n\n## Reference Material (provided by user)\n\n{req.reference.strip()}"

        user_msg = f"Topic: {req.name}\nLearning depth: {depth_profile['label']}{ref_section}"
        syllabus_content = _strip_markdown_fences(_call_llm(_build_syllabus_prompt(req.learning_depth), user_msg))
        syllabus = Syllabus(course_id=course.id, content=syllabus_content)
        db.add(syllabus)
        db.flush()

        prompt = FIRST_LESSON_PROMPT.format(syllabus=syllabus_content)
        lesson_user_msg = f"Generate the first lesson for topic '{req.name}' at '{depth_profile['label']}' learning depth{ref_section}"
        lesson_content = _strip_markdown_fences(_call_llm(prompt, lesson_user_msg))
        lesson = Lesson(course_id=course.id, number=1, content=lesson_content)
        db.add(lesson)

        _record_event(db, course.id, "course_created")
        _record_event(db, course.id, "lesson_generated", lesson_number=1)

        db.commit()
        db.refresh(course)

        return CourseDetailResponse(
            id=course.id, name=course.name, mode=course.mode, status=course.status,
            learning_depth=course.learning_depth, is_project=course.is_project,
            created_at=course.created_at, lesson_count=1,
            syllabus_content=syllabus_content,
            mastery_progress=0.0,
            source_filename=course.source_filename,
        )
    except Exception:
        logger.exception("Course creation error")
        db.rollback()
        raise HTTPException(status_code=500, detail="Course creation failed; please try again")


@router.post("/courses/from-source", response_model=CreateSourceCourseResponse)
async def create_course_from_source(
    name: str = Form(""),
    learning_depth: str = Form("standard"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Create a source-mode course from an uploaded PDF/TXT/Markdown file."""
    depth_profile = _learning_depth_profile(learning_depth)
    filename, source_text = await _extract_upload_text(file)
    course_name = name.strip() or filename.rsplit(".", 1)[0]

    course = Course(
        name=course_name,
        mode="source",
        status="learning",
        learning_depth=learning_depth,
        source_filename=filename,
        source_content=source_text,
    )
    db.add(course)
    db.flush()

    try:
        user_msg = f"""Topic: {course_name}
Learning depth: {depth_profile['label']}

Generate a course syllabus based on the user-uploaded source material below. The syllabus should serve understanding and mastering this material, not broadly discussing the topic by name.

## Full Source Material

{source_text}
"""
        syllabus_content = _strip_markdown_fences(_call_llm(_build_syllabus_prompt(learning_depth), user_msg))
        syllabus = Syllabus(course_id=course.id, content=syllabus_content)
        db.add(syllabus)

        source_lesson = Lesson(
            course_id=course.id,
            number=1,
            content=_source_lesson_content(filename, source_text),
            is_source=True,
        )
        db.add(source_lesson)

        _record_event(db, course.id, "source_course_created")
        _record_event(db, course.id, "source_lesson_created", lesson_number=1)

        db.commit()
        db.refresh(course)

        return CreateSourceCourseResponse(
            id=course.id,
            name=course.name,
            mode=course.mode,
            status=course.status,
            learning_depth=course.learning_depth,
            is_project=course.is_project,
            created_at=course.created_at,
            lesson_count=1,
            syllabus_content=syllabus_content,
            mastery_progress=0.0,
            source_filename=course.source_filename,
        )
    except Exception:
        logger.exception("Source course creation error")
        db.rollback()
        raise HTTPException(status_code=500, detail="Source course creation failed; please try again")


@router.post("/courses/from-project", response_model=CreateSourceCourseResponse)
async def create_course_from_project(
    name: str = Form(""),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """上传一个文件或整个文件夹作为「项目」：每个文件直接渲染成一篇，可随时划线提问；不生成大纲、不生成下一篇。"""
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one file")

    extracted: list[tuple[str, str, bytes | None]] = []
    for f in files:
        path = f.filename or "file"
        name_only = path.rsplit("/", 1)[-1]
        suffix = name_only.rsplit(".", 1)[-1].lower() if "." in name_only else ""
        raw = await f.read()
        if suffix == "pdf":
            try:
                content = _extract_pdf_text(raw)
            except Exception:
                content = ""
            extracted.append((path, content, raw))  # 原始 PDF 仍由前端 pdf.js 渲染；content 供划线问答上下文使用
            continue
        text = None
        for enc in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            continue
        if suffix in ("md", "markdown"):
            content = text
        else:
            lang = _CODE_LANGS.get(suffix, "")
            content = f"# {name_only}\n\n```{lang}\n{text}\n```"
        if content.strip():
            extracted.append((path, content, None))
    if not extracted:
        raise HTTPException(status_code=400, detail="No readable file content")

    extracted.sort(key=lambda item: item[0])  # 默认按文件路径字典序排列

    project_name = name.strip()
    if not project_name:
        first_path = extracted[0][0]
        if len(extracted) == 1:
            project_name = first_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        elif "/" in first_path:
            project_name = first_path.split("/", 1)[0]
        else:
            project_name = "Project"

    course = Course(
        name=project_name,
        mode="source",
        is_project=True,
        status="learning",
        learning_depth="standard",
        source_filename=extracted[0][0] if len(extracted) == 1 else None,
    )
    db.add(course)
    db.flush()

    for i, (path, content, blob) in enumerate(extracted, start=1):
        db.add(Lesson(
            course_id=course.id,
            number=i,
            content=content,
            is_source=True,
            source_filename=path,
            source_blob=blob,
        ))

    _record_event(db, course.id, "project_created")
    db.commit()
    db.refresh(course)

    return CreateSourceCourseResponse(
        id=course.id, name=course.name, mode=course.mode,
        status=course.status, learning_depth=course.learning_depth,
        is_project=course.is_project, created_at=course.created_at,
        lesson_count=len(extracted), syllabus_content=None,
        mastery_progress=0.0, source_filename=course.source_filename,
    )


@router.get("/courses/{course_id}/lessons/{lesson_num}/file")
def get_lesson_file(course_id: int, lesson_num: int, db: Session = Depends(get_db)):
    """返回某一篇的原始文件二进制（如 PDF），供前端 pdf.js 原生渲染。"""
    lesson = db.query(Lesson).join(Course).filter(
        Course.id == course_id, Lesson.number == lesson_num
    ).first()
    if not lesson or not lesson.source_blob:
        raise HTTPException(status_code=404, detail="This lesson has no source file")
    name = (lesson.source_filename or "file").rsplit("/", 1)[-1]
    media = "application/pdf" if name.lower().endswith(".pdf") else "application/octet-stream"
    return Response(content=lesson.source_blob, media_type=media)


@router.get("/courses", response_model=list[CourseResponse])
def list_courses(db: Session = Depends(get_db)):
    courses = db.query(Course).order_by(Course.created_at.desc()).all()
    return [
        CourseResponse(
            id=c.id, name=c.name, mode=c.mode, status=c.status,
            learning_depth=c.learning_depth, is_project=c.is_project,
            created_at=c.created_at, lesson_count=len(c.lessons),
            mastery_progress=_mastery_progress(c.syllabus.content) if c.syllabus else 0.0,
            source_filename=c.source_filename,
        )
        for c in courses
    ]


@router.delete("/courses/{course_id}")
def delete_course(course_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    db.delete(course)
    db.commit()
    return {"ok": True}


@router.get("/courses/{course_id}", response_model=CourseDetailResponse)
def get_course(course_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    progress = _mastery_progress(course.syllabus.content) if course.syllabus else 0.0
    return CourseDetailResponse(
        id=course.id, name=course.name, mode=course.mode, status=course.status,
        learning_depth=course.learning_depth, is_project=course.is_project,
        created_at=course.created_at, lesson_count=len(course.lessons),
        syllabus_content=course.syllabus.content if course.syllabus else None,
        mastery_progress=progress,
        source_filename=course.source_filename,
    )


@router.get("/courses/{course_id}/syllabus", response_model=SyllabusResponse)
def get_syllabus(course_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if not course.syllabus:
        raise HTTPException(status_code=404, detail="Syllabus not yet generated")
    return course.syllabus


@router.put("/courses/{course_id}/syllabus", response_model=SyllabusResponse)
def update_syllabus(
    course_id: int,
    req: SyllabusUpdateRequest,
    db: Session = Depends(get_db),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if not course.syllabus:
        raise HTTPException(status_code=404, detail="Syllabus not yet generated")
    course.syllabus.content = req.content
    db.commit()
    db.refresh(course.syllabus)
    return course.syllabus


def _extract_title(content: str) -> str:
    """Extract first H1 title from markdown content."""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return ""


@router.get("/courses/{course_id}/lessons", response_model=list[LessonListItem])
def list_lessons(course_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return [
        LessonListItem(
            id=lesson.id, number=lesson.number, is_evaluation=lesson.is_evaluation,
            is_source=lesson.is_source, source_filename=lesson.source_filename,
            title=_extract_title(lesson.content),
            has_feedback=lesson.feedback is not None,
            created_at=lesson.created_at,
        )
        for lesson in course.lessons
    ]


@router.get("/courses/{course_id}/lessons/{lesson_num}", response_model=LessonResponse)
def get_lesson(course_id: int, lesson_num: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return lesson


@router.post("/courses/{course_id}/lessons/{lesson_num}/annotations")
def create_annotation(
    course_id: int,
    lesson_num: int,
    req: CreateAnnotationRequest,
    db: Session = Depends(get_db),
):
    """Highlight a passage and ask → the answer streams back (SSE); session is saved on completion."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    system_prompt = _annotation_system_prompt(course, lesson, req.original_text)
    history = [{"role": "user", "content": req.comment}]
    lesson_id = lesson.id
    pos_start, pos_end, anchor_top = req.position_start, req.position_end, req.anchor_top
    original_text, comment = req.original_text, req.comment
    pdf_position = req.pdf_position
    cid, lnum = course_id, lesson_num

    def generate():
        try:
            answer = ""
            for content, is_final in _stream_llm_messages(system_prompt, history):
                if is_final:
                    answer = _strip_markdown_fences(content)
                else:
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            full_history = history + [{"role": "assistant", "content": answer}]
            annotation = Annotation(
                lesson_id=lesson_id,
                position_start=pos_start,
                position_end=pos_end,
                original_text=original_text,
                comment=comment,
                answer=answer,
                messages=json.dumps(full_history, ensure_ascii=False),
                anchor_top=anchor_top,
                pdf_position=pdf_position,
            )
            db.add(annotation)
            _record_event(db, cid, "annotation_answered", lesson_number=lnum)
            db.commit()
            db.refresh(annotation)
            payload = _annotation_to_response(annotation).model_dump(mode="json")
            yield f"data: {json.dumps({'done': True, 'annotation': payload}, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Annotation answer streaming error")
            db.rollback()
            yield f"data: {json.dumps({'error': 'Service temporarily unavailable; please try again'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/courses/{course_id}/lessons/{lesson_num}/annotations", response_model=list[AnnotationResponse])
def get_annotations(
    course_id: int,
    lesson_num: int,
    db: Session = Depends(get_db),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    annotations = db.query(Annotation).filter(Annotation.lesson_id == lesson.id).order_by(Annotation.created_at).all()
    return [_annotation_to_response(a) for a in annotations]


@router.post("/courses/{course_id}/lessons/{lesson_num}/annotations/{annotation_id}/messages")
def add_annotation_message(
    course_id: int,
    lesson_num: int,
    annotation_id: int,
    req: AddAnnotationMessageRequest,
    db: Session = Depends(get_db),
):
    """Follow-up question in an existing highlight session — the answer streams back (SSE)."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    annotation = db.query(Annotation).filter(
        Annotation.id == annotation_id, Annotation.lesson_id == lesson.id
    ).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    system_prompt = _annotation_system_prompt(course, lesson, annotation.original_text)
    history = _load_messages(annotation)
    history.append({"role": "user", "content": req.content})
    aid = annotation.id
    cid, lnum = course_id, lesson_num

    def generate():
        try:
            answer = ""
            for content, is_final in _stream_llm_messages(system_prompt, history):
                if is_final:
                    answer = _strip_markdown_fences(content)
                else:
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            full_history = history + [{"role": "assistant", "content": answer}]
            ann = db.query(Annotation).filter(Annotation.id == aid).first()
            ann.messages = json.dumps(full_history, ensure_ascii=False)
            ann.answer = answer  # keep latest answer in legacy column
            _record_event(db, cid, "annotation_answered", lesson_number=lnum)
            db.commit()
            db.refresh(ann)
            payload = _annotation_to_response(ann).model_dump(mode="json")
            yield f"data: {json.dumps({'done': True, 'annotation': payload}, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Annotation follow-up streaming error")
            db.rollback()
            yield f"data: {json.dumps({'error': 'Service temporarily unavailable; please try again'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/courses/{course_id}/lessons/{lesson_num}/annotations/save", response_model=AnnotationResponse)
def save_interrupted_annotation(
    course_id: int,
    lesson_num: int,
    req: SaveInterruptedRequest,
    db: Session = Depends(get_db),
):
    """用户中途点了「停止」：把已生成的部分回答落库（首次→新建批注，追问→追加到会话）。"""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    text = req.partial_answer.rstrip()
    if not text:
        raise HTTPException(status_code=400, detail="No content to save")

    if req.annotation_id is None:
        full_history = [
            {"role": "user", "content": req.comment},
            {"role": "assistant", "content": text},
        ]
        annotation = Annotation(
            lesson_id=lesson.id,
            position_start=req.position_start,
            position_end=req.position_end,
            original_text=req.original_text,
            comment=req.comment,
            answer=text,
            messages=json.dumps(full_history, ensure_ascii=False),
            anchor_top=req.anchor_top,
            pdf_position=req.pdf_position,
        )
        db.add(annotation)
    else:
        annotation = db.query(Annotation).filter(
            Annotation.id == req.annotation_id, Annotation.lesson_id == lesson.id
        ).first()
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")
        history = _load_messages(annotation)
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": text})
        annotation.messages = json.dumps(history, ensure_ascii=False)
        annotation.answer = text

    _record_event(db, course_id, "annotation_answered", lesson_number=lesson_num)
    db.commit()
    db.refresh(annotation)
    return _annotation_to_response(annotation)


@router.delete("/courses/{course_id}/lessons/{lesson_num}/annotations/{annotation_id}")
def delete_annotation(
    course_id: int,
    lesson_num: int,
    annotation_id: int,
    db: Session = Depends(get_db),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    annotation = db.query(Annotation).filter(
        Annotation.id == annotation_id, Annotation.lesson_id == lesson.id
    ).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    db.delete(annotation)
    db.commit()
    return {"ok": True}


@router.post("/courses/{course_id}/lessons/{lesson_num}/feedback", response_model=FeedbackResponse)
def create_feedback(
    course_id: int,
    lesson_num: int,
    req: CreateFeedbackRequest,
    db: Session = Depends(get_db),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    # Check if feedback already exists, update if so
    existing = db.query(Feedback).filter(Feedback.lesson_id == lesson.id).first()
    if existing:
        existing.content = req.content
        existing.thought_answers = req.thought_answers
        _record_event(db, course_id, "feedback_updated", lesson_number=lesson_num)
        db.commit()
        db.refresh(existing)
        return existing

    feedback = Feedback(
        lesson_id=lesson.id,
        content=req.content,
        thought_answers=req.thought_answers,
    )
    db.add(feedback)
    _record_event(db, course_id, "feedback_submitted", lesson_number=lesson_num)
    db.commit()
    db.refresh(feedback)
    return feedback


@router.post("/courses/{course_id}/next")
def generate_next_lesson(
    course_id: int,
    db: Session = Depends(get_db),
):
    """'我读完了' → AI generates next lesson (SSE streaming)."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.status == "completed":
        raise HTTPException(status_code=400, detail="Course already completed")
    if not course.syllabus:
        raise HTTPException(status_code=400, detail="Syllabus not yet generated")

    lessons = course.lessons
    if not lessons:
        raise HTTPException(status_code=400, detail="No lessons yet")

    last_lesson = lessons[-1]

    # Check if last lesson is an evaluation article — if so, generate summary
    if last_lesson.is_evaluation:
        return _generate_summary_response(course, db)

    syllabus_content = course.syllabus.content
    all_mastery_done = _check_all_mastery_items_done(syllabus_content)

    # Collect feedback and annotations from last lesson
    last_feedback = db.query(Feedback).filter(
        Feedback.lesson_id == last_lesson.id
    ).first()
    last_annotations = db.query(Annotation).filter(
        Annotation.lesson_id == last_lesson.id
    ).order_by(Annotation.created_at).all()

    feedback_text = ""
    if last_feedback:
        feedback_text = f"Feedback: {last_feedback.content}\nReflection question answers: {last_feedback.thought_answers}"
    else:
        feedback_text = "The student did not submit feedback."

    annotations_text = _format_annotations(last_annotations)

    next_number = last_lesson.number + 1

    # Extract the previous lesson's real thought questions as a dedicated field so
    # the review section can map to them exactly, instead of relying on the model
    # to spot them inside the (possibly truncated) full text (issue #3).
    last_questions = _extract_thought_questions(last_lesson.content) or \
        "(The previous lesson did not explicitly include a reflection questions section; please construct reasonable reflection questions from its content and review each one.)"

    if course.mode == "source" and last_lesson.is_source:
        prompt = SOURCE_LESSON_PROMPT.format(
            syllabus=syllabus_content,
            source_filename=course.source_filename or "uploaded-source",
            source_content=course.source_content or last_lesson.content,
            source_annotations=_format_annotations(last_annotations),
            lesson_number=next_number,
        )
        user_msg = f"Generate lesson {next_number} based on the source material and highlight Q&A"
    elif all_mastery_done:
        prompt = EVAL_LESSON_PROMPT.format(
            syllabus=syllabus_content,
            last_lesson=last_lesson.content,
            feedback=feedback_text,
            annotations=annotations_text,
            last_questions=last_questions,
        )
        user_msg = "Generate the evaluation lesson"
    else:
        recent = lessons[-3:] if len(lessons) > 3 else lessons
        prev_text = "\n\n---\n\n".join(
            f"### Lesson {lesson.number}\n{lesson.content[:20000]}" for lesson in recent
        )
        prompt = NEXT_LESSON_PROMPT.format(
            syllabus=syllabus_content,
            previous_lessons=prev_text,
            feedback=feedback_text,
            annotations=annotations_text,
            lesson_number=next_number,
            last_questions=last_questions,
        )
        user_msg = f"Generate lesson {next_number}"

    cid = course.id

    def generate():
        try:
            lesson_content = ""
            for content, is_final in _stream_llm(prompt, user_msg):
                if is_final:
                    lesson_content = content
                else:
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            lesson_content = _strip_markdown_fences(lesson_content)
            is_eval = lesson_content.strip().startswith("<!-- eval-article -->")

            # Use request db session (alive during streaming)
            new_lesson = Lesson(
                course_id=cid, number=next_number,
                content=lesson_content, is_evaluation=is_eval,
            )
            db.add(new_lesson)
            _record_event(db, cid, "lesson_generated", lesson_number=next_number)

            # Auto-update syllabus mastery items based on lesson content
            if not is_eval:
                course_obj = db.query(Course).filter(Course.id == cid).first()
                if course_obj and course_obj.syllabus:
                    updated = _auto_check_mastery(course_obj.syllabus.content, lesson_content)
                    if updated != course_obj.syllabus.content:
                        course_obj.syllabus.content = updated

            db.commit()

            yield f"data: {json.dumps({'done': True, 'lesson_number': next_number, 'is_evaluation': is_eval}, ensure_ascii=False)}\n\n"

        except Exception:
            logger.exception("Next lesson generation error")
            db.rollback()
            yield f"data: {json.dumps({'error': 'Service temporarily unavailable; please try again'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _generate_summary_response(course: Course, db: Session):
    """Generate summary after evaluation article is read."""
    syllabus_content = course.syllabus.content
    all_lessons_text = "\n\n---\n\n".join(
        f"### Lesson {lesson.number}\n{lesson.content}" for lesson in course.lessons
    )

    cid = course.id

    def generate():
        try:
            prompt = SUMMARY_PROMPT.format(
                syllabus=syllabus_content,
                all_lessons=all_lessons_text,
            )
            summary_content = ""
            for content, is_final in _stream_llm(prompt, "Generate the course summary"):
                if is_final:
                    summary_content = content
                else:
                    yield f"data: {json.dumps({'phase': 'summary', 'content': content}, ensure_ascii=False)}\n\n"

            summary_content = _strip_markdown_fences(summary_content)
            # Save summary as lesson number=0 and mark course completed
            course_obj = db.query(Course).filter(Course.id == cid).first()
            course_obj.status = "completed"
            summary_lesson = Lesson(
                course_id=cid, number=0, content=summary_content, is_evaluation=False,
            )
            db.add(summary_lesson)
            db.commit()

            yield f"data: {json.dumps({'done': True, 'completed': True}, ensure_ascii=False)}\n\n"

        except Exception:
            logger.exception("Summary generation error")
            db.rollback()
            yield f"data: {json.dumps({'error': 'Summary generation failed; please try again'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/courses/{course_id}/summary")
def get_summary(course_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    # Summary is stored as lesson number 0
    summary = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == 0).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not yet generated")
    return {"content": summary.content}


# ---------------------------------------------------------------------------
# Feedback GET — restore saved feedback on page load
# ---------------------------------------------------------------------------

@router.get("/courses/{course_id}/lessons/{lesson_num}/feedback")
def get_feedback(course_id: int, lesson_num: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lesson = db.query(Lesson).filter(Lesson.course_id == course_id, Lesson.number == lesson_num).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    feedback = db.query(Feedback).filter(Feedback.lesson_id == lesson.id).first()
    if not feedback:
        return {"exists": False, "content": "", "thought_answers": ""}
    return {
        "exists": True,
        "id": feedback.id,
        "content": feedback.content,
        "thought_answers": feedback.thought_answers or "",
        "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
    }


# ---------------------------------------------------------------------------
# Lesson opened event — frontend calls this when user opens a lesson
# ---------------------------------------------------------------------------

@router.post("/courses/{course_id}/lessons/{lesson_num}/opened")
def record_lesson_opened(course_id: int, lesson_num: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    _record_event(db, course_id, "lesson_opened", lesson_number=lesson_num)
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Learning Stats
# ---------------------------------------------------------------------------

@router.get("/courses/{course_id}/stats", response_model=CourseStatsResponse)
def get_course_stats(course_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    normal_lessons = [lesson for lesson in course.lessons if lesson.number > 0]
    total_annotations = sum(len(lesson.annotations) for lesson in normal_lessons)
    total_feedback = sum(1 for lesson in normal_lessons if lesson.feedback)

    checked, total = (0, 0)
    if course.syllabus:
        checked, total = _count_mastery_items(course.syllabus.content)

    events = db.query(LearningEvent).filter(
        LearningEvent.course_id == course_id
    ).order_by(LearningEvent.created_at).all()

    first_activity = events[0].created_at if events else None
    last_activity = events[-1].created_at if events else None

    return CourseStatsResponse(
        total_lessons=len(normal_lessons),
        total_annotations=total_annotations,
        total_feedback=total_feedback,
        mastery_checked=checked,
        mastery_total=total,
        mastery_progress=checked / total if total > 0 else 0.0,
        first_activity=first_activity,
        last_activity=last_activity,
    )


def _event_local_date(dt: datetime) -> date:
    """把事件时间戳归一到服务器本地日期。

    SQLite 取回的是裸时间（无时区，实为 UTC），先补 UTC 再转本地，
    避免凌晨学习被错算到「昨天」。tz-aware 的输入也能正确处理。
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().date()


@router.get("/stats", response_model=GlobalStatsResponse)
def get_global_stats(db: Session = Depends(get_db)):
    courses = db.query(Course).all()
    total_courses = len(courses)
    active_courses = sum(1 for c in courses if c.status == "learning")
    completed_courses = sum(1 for c in courses if c.status == "completed")

    total_lessons = db.query(Lesson).filter(Lesson.number > 0).count()
    total_annotations = db.query(Annotation).count()
    total_feedback = db.query(Feedback).count()

    # Calculate streaks from learning events（按本地日期，与学习日历口径一致）
    raw_times = db.query(LearningEvent.created_at).all()
    event_dates = sorted(
        {_event_local_date(r[0]) for r in raw_times if r[0]}, reverse=True
    )

    current_streak = 0
    longest_streak = 0

    if event_dates:
        # Current streak: count consecutive days ending today or yesterday
        today = date.today()
        streak = 0
        for i, d in enumerate(event_dates):
            if isinstance(d, str):
                d = date.fromisoformat(d)
            expected = today - timedelta(days=i)
            if d == expected:
                streak += 1
            elif i == 0 and d == today - timedelta(days=1):
                # Allow streak to start from yesterday
                streak += 1
                today = today - timedelta(days=1)
            else:
                break
        current_streak = streak

        # Longest streak
        if event_dates:
            sorted_dates = sorted(set(date.fromisoformat(str(d)) if isinstance(d, str) else d for d in event_dates))
            best = 1
            run = 1
            for i in range(1, len(sorted_dates)):
                if sorted_dates[i] - sorted_dates[i - 1] == timedelta(days=1):
                    run += 1
                    best = max(best, run)
                else:
                    run = 1
            longest_streak = best

    return GlobalStatsResponse(
        total_courses=total_courses,
        active_courses=active_courses,
        completed_courses=completed_courses,
        total_lessons_read=total_lessons,
        total_annotations=total_annotations,
        total_feedback=total_feedback,
        current_streak=current_streak,
        longest_streak=longest_streak,
    )


@router.get("/calendar", response_model=CalendarResponse)
def get_learning_calendar(db: Session = Depends(get_db)):
    """按天聚合所有学习活动，供个人中心的学习日历展示「哪天学了什么」。

    日期归组用本地日期（与 /stats 的连续天数口径一致），每天给出：
    接触到的课程、对应课文编号、划线条数与事件总数。
    划线按 Annotation 行数（真实划线条数）统计，而非问答事件数——
    避免追问轮次把数字撑大，从而与 /stats 的 total_annotations 一致。
    """
    course_meta = {c.id: (c.name, c.mode) for c in db.query(Course).all()}

    def _cell(course_id: int) -> dict:
        name, mode = course_meta.get(course_id, ("", "topic"))
        return {
            "course_id": course_id,
            "course_name": name,
            "mode": mode,
            "lessons": set(),
            "annotations": 0,
            "event_count": 0,
        }

    days: dict[str, dict] = {}

    # 课文与事件：从 learning_events 取课文编号与事件总数
    for event in db.query(LearningEvent).order_by(LearningEvent.created_at).all():
        if not event.created_at or event.course_id not in course_meta:
            continue
        day_key = _event_local_date(event.created_at).isoformat()
        c = days.setdefault(day_key, {}).setdefault(event.course_id, _cell(event.course_id))
        c["event_count"] += 1
        if event.lesson_number and event.lesson_number > 0:
            c["lessons"].add(event.lesson_number)

    # 划线：从 annotations 表按创建日期统计真实划线条数（追问不重复计）
    ann_rows = (
        db.query(Annotation.created_at, Lesson.course_id)
        .join(Lesson, Annotation.lesson_id == Lesson.id)
        .all()
    )
    for created_at, course_id in ann_rows:
        if not created_at or course_id not in course_meta:
            continue
        day_key = _event_local_date(created_at).isoformat()
        c = days.setdefault(day_key, {}).setdefault(course_id, _cell(course_id))
        c["annotations"] += 1

    result_days = []
    for day_key in sorted(days.keys()):
        course_list, lessons_read, annotations, event_count = [], 0, 0, 0
        for c in days[day_key].values():
            lessons_sorted = sorted(c["lessons"])
            lessons_read += len(lessons_sorted)
            annotations += c["annotations"]
            event_count += c["event_count"]
            course_list.append(CalendarCourseActivity(
                course_id=c["course_id"],
                course_name=c["course_name"],
                mode=c["mode"],
                lessons=lessons_sorted,
                annotations=c["annotations"],
                event_count=c["event_count"],
            ))
        course_list.sort(key=lambda x: x.event_count, reverse=True)
        result_days.append(CalendarDay(
            date=day_key,
            event_count=event_count,
            lessons_read=lessons_read,
            annotations=annotations,
            courses=course_list,
        ))

    return CalendarResponse(
        days=result_days,
        total_active_days=len(result_days),
        first_active_date=result_days[0].date if result_days else None,
        last_active_date=result_days[-1].date if result_days else None,
    )
