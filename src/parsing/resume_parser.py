import json
from typing import List, Optional
from pydantic import BaseModel, Field
import google.generativeai as genai

from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL

genai.configure(api_key=GOOGLE_API_KEY)


# --- Schema definition ---

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


# --- Gemini-compatible schema builder ---

def _clean_schema(schema, defs: dict):
    """
    Recursively convert a Pydantic-generated JSON schema into a Gemini-compatible
    schema: inline $ref pointers, collapse anyOf/Optional unions, and strip
    metadata keys Gemini doesn't support (default, title) — WITHOUT touching
    actual property names inside "properties" dicts.
    """
    if isinstance(schema, list):
        return [_clean_schema(item, defs) for item in schema]

    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        return _clean_schema(defs.get(ref_name, {}), defs)

    if "anyOf" in schema:
        # Pydantic represents Optional[X] as anyOf: [{X schema}, {"type": "null"}]
        # Gemini doesn't support anyOf, so pick the first non-null branch.
        non_null_options = [opt for opt in schema["anyOf"] if opt.get("type") != "null"]
        chosen = non_null_options[0] if non_null_options else {"type": "string"}
        resolved = _clean_schema(chosen, defs)
        if "description" in schema and "description" not in resolved:
            resolved["description"] = schema["description"]
        return resolved

    cleaned = {}
    for key, value in schema.items():
        if key == "default":
            continue
        if key == "title":
            # Only drop "title" when it's schema metadata (a string value),
            # never when it's the "properties" dict containing a field named "title".
            if isinstance(value, str):
                continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop_name: _clean_schema(prop_schema, defs)
                for prop_name, prop_schema in value.items()
            }
        else:
            cleaned[key] = _clean_schema(value, defs)
    return cleaned


def build_gemini_schema(model_cls: type[BaseModel]) -> dict:
    """Convert a Pydantic model into a schema Gemini's API can accept."""
    raw_schema = model_cls.model_json_schema()
    defs = raw_schema.pop("$defs", {})
    return _clean_schema(raw_schema, defs)


# --- Parsing function ---

SYSTEM_PROMPT = """You are a precise resume-parsing assistant.
Extract structured information from the resume text provided.
Rules:
- Only extract information that is actually present in the resume. Do not invent details.
- If a field is not found, leave it empty or null.
- For 'target_role', infer the most likely role based on skills and experience, but only if there is reasonable evidence.
- Return ONLY the structured data matching the required schema."""

_GEMINI_SCHEMA = build_gemini_schema(ResumeProfile)


def parse_resume(resume_text: str) -> ResumeProfile:
    """
    Send resume text to Gemini and return a validated ResumeProfile object.
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_CHAT_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(
        f"Resume text:\n\n{resume_text}",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_GEMINI_SCHEMA,
        ),
    )

    raw_json = response.text

    try:
        data = json.loads(raw_json)
        profile = ResumeProfile.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM output into ResumeProfile: {e}\nRaw output: {raw_json}")

    return profile