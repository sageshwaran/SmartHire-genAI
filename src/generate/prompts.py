"""
Prompt library for SmartHire GenAI.

Each prompt is defined as a function that returns a fully-formed prompt string,
so callers never string-format raw templates themselves. Keeping prompts here
(rather than inline in cv_suggestions.py) makes them independently testable
and easy to version/compare during evaluation (Phase 7).
"""


def cv_improvement_prompt_v1(resume_summary: str, target_job_description: str) -> str:
    """
    v1: baseline prompt — straightforward instruction, no explicit reasoning steps.
    Kept for before/after comparison during evaluation.
    """
    return f"""You are a career coach. Given the candidate's resume and a target job description,
suggest improvements to their CV.

Candidate resume:
{resume_summary}

Target job description:
{target_job_description}

List missing skills, weak points, and suggest a better summary."""


def cv_improvement_prompt_v2(resume_summary: str, target_job_description: str) -> str:
    """
    v2: improved prompt — explicit structure, grounding instructions, and
    a request for specificity over generic advice. This is the production version.
    """
    return f"""You are an expert technical career coach helping a candidate tailor their
resume for a specific job.

CANDIDATE RESUME:
{resume_summary}

TARGET JOB DESCRIPTION:
{target_job_description}

Your task: compare the resume against the job description and produce specific,
actionable feedback. Follow these rules:
- Base every suggestion on an actual gap or mismatch between the resume and the
  job description. Do not give generic advice that could apply to any resume.
- For "missing_skills": only list skills that appear in the job description but
  are absent from the resume's skills or experience.
- For "weak_bullet_points": identify specific existing resume lines that are vague
  or lack measurable impact, and explain briefly why each is weak.
- For "rewritten_summary": write a 2-3 sentence professional summary that
  authentically reflects the candidate's actual background while emphasizing
  the overlap with the target role. Do not invent experience the candidate
  does not have.

Return your response matching the required structured schema."""