import json
from typing import List
from pydantic import BaseModel, Field
import google.generativeai as genai

from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL
from src.parsing.resume_parser import ResumeProfile, build_gemini_schema
from src.generate.prompts import cv_improvement_prompt_v2

genai.configure(api_key=GOOGLE_API_KEY)


# --- Schema definition ---

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


_GEMINI_SCHEMA = build_gemini_schema(CVSuggestions)


# --- Helper: turn a ResumeProfile into a compact text block for the prompt ---

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


# --- Main function ---

def generate_cv_suggestions(profile: ResumeProfile, target_job_description: str) -> CVSuggestions:
    """
    Given a parsed resume profile and a target job description, generate
    specific improvement suggestions using Gemini structured output.
    """
    resume_summary = _resume_to_summary_text(profile)
    prompt = cv_improvement_prompt_v2(resume_summary, target_job_description)

    model = genai.GenerativeModel(model_name=GEMINI_CHAT_MODEL)

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_GEMINI_SCHEMA,
        ),
    )

    raw_json = response.text

    try:
        data = json.loads(raw_json)
        suggestions = CVSuggestions.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM output into CVSuggestions: {e}\nRaw output: {raw_json}")

    return suggestions