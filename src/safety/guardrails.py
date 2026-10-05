import re
import json
from enum import Enum
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL

client = genai.Client(api_key=GOOGLE_API_KEY)

MAX_INPUT_LENGTH = 1000           # for mentor chat questions
MAX_DOCUMENT_LENGTH = 8000        # for resumes / job descriptions

_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) instructions",
    r"disregard (all )?(previous|above|prior) instructions",
    r"you are now",
    r"forget (all )?(previous|your) instructions",
    r"system prompt",
    r"new instructions:",
    r"act as (a |an )?(?!career|mentor)",
]


class RejectionReason(str, Enum):
    EMPTY = "empty_input"
    TOO_LONG = "input_too_long"
    INJECTION_ATTEMPT = "injection_attempt"
    OFF_TOPIC = "off_topic"


class GuardrailResult(BaseModel):
    is_allowed: bool
    reason: RejectionReason | None = None
    message: str = Field(default="", description="A friendly, user-facing explanation if rejected")


def _rule_based_check(text: str, max_length: int = MAX_INPUT_LENGTH) -> GuardrailResult | None:
    stripped = text.strip()

    if not stripped:
        return GuardrailResult(
            is_allowed=False,
            reason=RejectionReason.EMPTY,
            message="Please enter a question.",
        )

    if len(stripped) > max_length:
        return GuardrailResult(
            is_allowed=False,
            reason=RejectionReason.TOO_LONG,
            message=f"Input exceeds the {max_length} character limit.",
        )

    lowered = stripped.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                is_allowed=False,
                reason=RejectionReason.INJECTION_ATTEMPT,
                message="Input rejected: contains a disallowed instruction pattern.",
            )

    return None

class TopicClassification(BaseModel):
    is_career_related: bool = Field(
        description="True if the question is about careers, jobs, skills, interviews, or resumes"
    )


_CLASSIFIER_PROMPT = """Classify whether the following user question is related to careers, \
job searching, skill development, resumes, or interview preparation.

Question: {question}

Respond with the classification only."""


def _llm_topic_check(text: str) -> GuardrailResult:
    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=_CLASSIFIER_PROMPT.format(question=text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TopicClassification,
        ),
    )

    data = json.loads(response.text)
    classification = TopicClassification.model_validate(data)

    if not classification.is_career_related:
        return GuardrailResult(
            is_allowed=False,
            reason=RejectionReason.OFF_TOPIC,
            message="I can only help with career, job search, and skill development topics. "
                    "Try asking something like 'How do I prepare for a Data Analyst interview?'",
        )

    return GuardrailResult(is_allowed=True)


def check_input(text: str) -> GuardrailResult:
    rule_result = _rule_based_check(text)
    if rule_result is not None:
        return rule_result

    return _llm_topic_check(text)