import json
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.safety.guardrails import _rule_based_check, MAX_DOCUMENT_LENGTH
from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL
from src.parsing.resume_parser import ResumeProfile
from src.generate.prompts import cv_improvement_prompt_v2

client = genai.Client(api_key=GOOGLE_API_KEY)


class WeakBulletPoint(BaseModel):
    original_line: str = Field(description="The resume line that is weak or vague")
    issue: str = Field(description="Why this line is weak (e.g. no measurable impact, too vague)")
    suggested_rewrite: str = Field(description="A stronger version of this line")


class CVSuggestions(BaseModel):
    missing_skills: List[str] = Field(
        description="Skills present in the job description but absent from the resume"
    )
    weak_bullet_points: List[WeakBulletPoint] = Field(
        default_factory=list,
        description="Specific resume lines that are vague or lack measurable impact"
    )
    rewritten_summary: str = Field(
        description="A 2-3 sentence professional summary tailored to the target role"
    )


def _resume_to_summary_text(profile: ResumeProfile) -> str:
    lines = [f"Name: {profile.name}"]
    if profile.summary:
        lines.append(f"Summary: {profile.summary}")
    lines.append(f"Skills: {', '.join(profile.skills)}")

    if profile.experience:
        lines.append("Experience:")
        for exp in profile.experience:
            desc = f" - {exp.description}" if exp.description else ""
            lines.append(f"  - {exp.title} at {exp.company} ({exp.duration}){desc}")

    if profile.education:
        lines.append("Education:")
        for edu in profile.education:
            lines.append(f"  - {edu.degree}, {edu.institution} ({edu.year or 'n/a'})")

    return "\n".join(lines)


def generate_cv_suggestions(profile: ResumeProfile, target_job_description: str) -> CVSuggestions:
    guard_result = _rule_based_check(target_job_description, max_length=MAX_DOCUMENT_LENGTH)
    if guard_result is not None:
        raise ValueError(f"Job description rejected by guardrails: {guard_result.message}")

    resume_summary = _resume_to_summary_text(profile)
    resume_summary = _resume_to_summary_text(profile)
    prompt = cv_improvement_prompt_v2(resume_summary, target_job_description)

    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CVSuggestions,
        ),
    )

    raw_json = response.text

    try:
        data = json.loads(raw_json)
        suggestions = CVSuggestions.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM output into CVSuggestions: {e}\nRaw output: {raw_json}")

    return suggestions