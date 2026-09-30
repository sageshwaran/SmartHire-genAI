import json
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL

client = genai.Client(api_key=GOOGLE_API_KEY)


# --- Schema definition (unchanged) ---

class Experience(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    duration: str = Field(description="e.g. 'Jan 2023 - Present' or '6 months'")
    description: Optional[str] = Field(default=None, description="Brief summary of role")


class Education(BaseModel):
    degree: str = Field(description="e.g. 'B.Tech in Computer Science'")
    institution: str
    year: Optional[str] = Field(default=None, description="Graduation year or expected year")


class ResumeProfile(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(description="List of technical and soft skills")
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    target_role: Optional[str] = Field(
        default=None,
        description="The role this person appears to be targeting, inferred from resume content"
    )
    summary: Optional[str] = Field(
        default=None,
        description="A 2-3 sentence professional summary of the candidate"
    )


SYSTEM_PROMPT = """You are a precise resume-parsing assistant.
Extract structured information from the resume text provided.
Rules:
- Only extract information that is actually present in the resume. Do not invent details.
- If a field is not found, leave it empty or null.
- For 'target_role', infer the most likely role based on skills and experience, but only if there is reasonable evidence.
- Return ONLY the structured data matching the required schema."""


def parse_resume(resume_text: str) -> ResumeProfile:
    """
    Send resume text to Gemini and return a validated ResumeProfile object.
    Uses the new google-genai SDK's native Pydantic schema support.
    """
    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=f"Resume text:\n\n{resume_text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ResumeProfile,
        ),
    )

    raw_json = response.text

    try:
        data = json.loads(raw_json)
        profile = ResumeProfile.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM output into ResumeProfile: {e}\nRaw output: {raw_json}")

    return profile