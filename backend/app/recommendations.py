import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.courses import _call_llm, _extract_title, _mastery_progress, _strip_markdown_fences
from app.database import get_db
from app.models import Course, LearningRecommendation
from app.schemas import (
    RecommendationDashboardResponse,
    RecommendationResponse,
    StartRecommendationRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

RECOMMENDATION_PROMPT = """You are the learning path recommendation module of the Bloom learning system.

Based on all courses the user has already studied, syllabus mastery items, lesson titles, and the to-learn list, recommend 3 next learning topics.

Recommendation principles:
1. Topics must grow bottom-up from what the user has already learned; explain which prior knowledge each builds on
2. Do not recommend topics the user has already studied, is currently studying, has saved, or has started
3. The three topics must be clearly distinct: one reinforces foundations, one makes cross-connections, one extends slightly outward
4. Topic names should be suitable for creating a course directly; do not write them as sentences
5. Output only a JSON array; do not output markdown or explanations

JSON format:
[
  {
    "title": "topic name",
    "rationale": "why to study it now, 1-2 sentences",
    "bridge": "how it connects to what the user has already learned, 1 sentence",
    "source_topics": ["learned topic A", "learned topic B"]
  }
]
"""

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", "", title.strip().lower())


def _decode_topics(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item).strip()]


def _to_response(item: LearningRecommendation) -> RecommendationResponse:
    return RecommendationResponse(
        id=item.id,
        title=item.title,
        rationale=item.rationale,
        bridge=item.bridge or "",
        source_topics=_decode_topics(item.source_topics),
        status=item.status,
        generation=item.generation,
        course_id=item.course_id,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _dashboard_response(db: Session) -> RecommendationDashboardResponse:
    suggestions = db.query(LearningRecommendation).filter(
        LearningRecommendation.status == "suggested"
    ).order_by(
        LearningRecommendation.generation.desc(),
        LearningRecommendation.created_at.desc(),
    ).limit(3).all()
    suggestions = list(reversed(suggestions))

    saved = db.query(LearningRecommendation).filter(
        LearningRecommendation.status == "saved"
    ).order_by(LearningRecommendation.updated_at.desc()).all()

    return RecommendationDashboardResponse(
        recommendations=[_to_response(item) for item in suggestions],
        saved=[_to_response(item) for item in saved],
    )


def _mastery_lines(course: Course) -> list[str]:
    if not course.syllabus:
        return []
    lines = []
    for line in course.syllabus.content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ["):
            lines.append(stripped)
    return lines[:10]


def _build_learning_profile(db: Session) -> str:
    courses = db.query(Course).order_by(Course.created_at.asc()).all()
    if not courses:
        return "The user has no course history yet. Please recommend introductory topics suitable as a foundation for long-term learning."

    blocks = []
    for course in courses:
        if course.is_project:
            blocks.append(_project_profile_block(course))
            continue
        lesson_titles = [
            _extract_title(lesson.content) or f"Lesson {lesson.number}"
            for lesson in course.lessons
            if lesson.number > 0
        ][:8]
        mastery = "\n".join(f"    {line}" for line in _mastery_lines(course)) or "    No syllabus mastery items yet"
        blocks.append(
            "\n".join([
                f"- Course: {course.name}",
                f"  Status: {course.status}",
                f"  Progress: {round(_mastery_progress(course.syllabus.content) * 100) if course.syllabus else 0}%",
                f"  Lessons: {', '.join(lesson_titles) if lesson_titles else 'No lessons yet'}",
                "  Mastery items:",
                mastery,
            ])
        )
    return "\n\n".join(blocks)


def _project_profile_block(course: Course) -> str:
    """项目课程的学习画像：文件清单 + 划线提问（用户在项目中的关注点与困惑）。"""
    files = [
        lesson.source_filename or _extract_title(lesson.content) or f"File {lesson.number}"
        for lesson in course.lessons
        if lesson.number > 0
    ][:12]
    questions = []
    for lesson in course.lessons:
        for ann in lesson.annotations:
            if ann.comment and ann.comment.strip():
                questions.append(ann.comment.strip())
    questions = questions[:10]
    parts = [
        f"- Project: {course.name} (files/code uploaded by user for study)",
        f"  Files: {', '.join(files) if files else 'No files yet'}",
    ]
    if questions:
        parts.append("  Highlight questions (user's focus areas and confusion in the project):")
        parts.extend(f"    - {q}" for q in questions)
    return "\n".join(parts)


def _avoid_titles(db: Session) -> list[str]:
    course_titles = [row[0] for row in db.query(Course.name).all()]
    recommendation_titles = [
        row[0]
        for row in db.query(LearningRecommendation.title).filter(
            LearningRecommendation.status.in_(("suggested", "saved", "started", "dismissed"))
        ).order_by(LearningRecommendation.created_at.desc()).limit(30).all()
    ]
    return [title for title in [*course_titles, *recommendation_titles] if title]


def _parse_recommendation_payload(raw: str) -> list[dict]:
    text = _strip_markdown_fences(raw)
    try:
        data = json.loads(text)
    except ValueError:
        match = _JSON_ARRAY_RE.search(text)
        if not match:
            raise ValueError("no JSON array found")
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise ValueError("recommendation payload must be a list")
    return [item for item in data if isinstance(item, dict)]


def _generate_recommendations(db: Session, generation: int) -> list[LearningRecommendation]:
    avoid = _avoid_titles(db)
    blocked = {_normalize_title(title) for title in avoid}
    user_message = f"""## Learned Content Profile

{_build_learning_profile(db)}

## Do Not Recommend These Topics

{", ".join(avoid) if avoid else "None"}
"""
    raw = _call_llm(RECOMMENDATION_PROMPT, user_message)

    try:
        payload = _parse_recommendation_payload(raw)
    except ValueError as exc:
        logger.exception("Recommendation payload parse error")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: could not parse model response ({exc})")

    seen = set()
    recommendations = []
    for item in payload:
        title = str(item.get("title", "")).strip()
        norm = _normalize_title(title)
        if not title or norm in blocked or norm in seen:
            continue
        seen.add(norm)
        topics = item.get("source_topics", [])
        if not isinstance(topics, list):
            topics = []
        recommendations.append(
            LearningRecommendation(
                title=title[:200],
                rationale=str(item.get("rationale", "")).strip()[:1000],
                bridge=str(item.get("bridge", "")).strip()[:1000],
                source_topics=json.dumps([str(topic) for topic in topics[:6]], ensure_ascii=False),
                status="suggested",
                generation=generation,
            )
        )
        if len(recommendations) == 3:
            break

    if len(recommendations) < 3:
        raise HTTPException(status_code=500, detail="Recommendation generation failed: fewer than 3 valid topics; please refresh and try again")
    return recommendations


@router.get("", response_model=RecommendationDashboardResponse)
def get_recommendations(db: Session = Depends(get_db)):
    return _dashboard_response(db)


@router.post("/refresh", response_model=RecommendationDashboardResponse)
def refresh_recommendations(db: Session = Depends(get_db)):
    generation = (db.query(func.max(LearningRecommendation.generation)).scalar() or 0) + 1
    new_items = _generate_recommendations(db, generation)

    db.query(LearningRecommendation).filter(
        LearningRecommendation.status == "suggested"
    ).update({"status": "dismissed"})
    db.add_all(new_items)
    db.commit()
    return _dashboard_response(db)


@router.post("/{recommendation_id}/save", response_model=RecommendationResponse)
def save_recommendation(recommendation_id: int, db: Session = Depends(get_db)):
    recommendation = db.query(LearningRecommendation).filter(
        LearningRecommendation.id == recommendation_id
    ).first()
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    if recommendation.status == "started":
        raise HTTPException(status_code=400, detail="Cannot add a recommendation you've already started to the to-learn list")
    recommendation.status = "saved"
    db.commit()
    db.refresh(recommendation)
    return _to_response(recommendation)


@router.delete("/{recommendation_id}/save")
def remove_saved_recommendation(recommendation_id: int, db: Session = Depends(get_db)):
    recommendation = db.query(LearningRecommendation).filter(
        LearningRecommendation.id == recommendation_id
    ).first()
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    recommendation.status = "dismissed"
    db.commit()
    return {"ok": True}


@router.post("/{recommendation_id}/start", response_model=RecommendationResponse)
def start_recommendation(
    recommendation_id: int,
    req: StartRecommendationRequest,
    db: Session = Depends(get_db),
):
    recommendation = db.query(LearningRecommendation).filter(
        LearningRecommendation.id == recommendation_id
    ).first()
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    if req.course_id:
        course = db.query(Course).filter(Course.id == req.course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
    recommendation.status = "started"
    recommendation.course_id = req.course_id
    db.commit()
    db.refresh(recommendation)
    return _to_response(recommendation)
