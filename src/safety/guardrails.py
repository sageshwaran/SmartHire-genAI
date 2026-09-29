import re
from enum import Enum
from pydantic import BaseModel, Field
import google.generativeai as genai

from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL
from src.parsing.resume_parser import build_gemini_schema  # reusing our schema cleaner

genai.configure(api_key=GOOGLE_API_KEY)

MAX_INPUT_LENGTH = 1000

# Patterns commonly used to try to override a system prompt
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) instructions",
    r"disregard (all )?(previous|above|prior) instructions",
    r"you are now",
    r"forget (all )?(previous|your) instructions",
    r"system prompt",
    r"new instructions:",
    r"act as (a |an )?(?!career|mentor)",  # "act as X" where X isn't career/mentor-related
]


class RejectionReason(str, Enum):
    EMPTY = "empty_input"
    TOO_LONG = "input_too_long"
    INJECTION_ATTEMPT = "injection_attempt"
    OFF_TOPIC = "off_topic"


class GuardrailResult(BaseModel):
    is_allowed: bool
    reason: RejectionReason | None = None
    message: str = Field(
        default="",
        description="A friendly, user-facing explanation if rejected"
    )

def _rule_based_check(text: str) -> GuardrailResult | None:
    """Fast, free checks. Returns None if the input passes (needs further checking)."""
    stripped = text.strip()

    if not stripped:
        return GuardrailResult(
            is_allowed=False,
            reason=RejectionReason.EMPTY,
            message="Please enter a question.",
        )

    if len(stripped) > MAX_INPUT_LENGTH:
        return GuardrailResult(
            is_allowed=False,
            reason=RejectionReason.TOO_LONG,
            message=f"Please keep your question under {MAX_INPUT_LENGTH} characters.",
        )

    lowered = stripped.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                is_allowed=False,
                reason=RejectionReason.INJECTION_ATTEMPT,
                message="I can't process that request. Please ask a career-related question.",
            )

    return None  # passed rule-based checks


# --- LLM-based topic classifier ---

class TopicClassification(BaseModel):
    is_career_related: bool = Field(
        description="True if the question is about careers, jobs, skills, interviews, or resumes"
    )


_CLASSIFIER_SCHEMA = build_gemini_schema(TopicClassification)

_CLASSIFIER_PROMPT = """Classify whether the following user question is related to careers, \
job searching, skill development, resumes, or interview preparation.

Question: {question}

Respond with the classification only."""


def _llm_topic_check(text: str) -> GuardrailResult:
    model = genai.GenerativeModel(model_name=GEMINI_CHAT_MODEL)
    response = model.generate_content(
        _CLASSIFIER_PROMPT.format(question=text),
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_CLASSIFIER_SCHEMA,
        ),
    )

    import json
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
    """
    Run the full guardrails pipeline on a user input: rule-based checks first
    (fast, free), then an LLM-based topic classifier if those pass.
    """
    rule_result = _rule_based_check(text)
    if rule_result is not None:
        return rule_result

    return _llm_topic_check(text)