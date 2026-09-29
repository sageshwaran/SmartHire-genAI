from src.parsing.resume_parser import ResumeProfile


def profile_to_search_text(profile: ResumeProfile) -> str:
    """
    Convert a parsed resume profile into a focused text block optimized for
    semantic search against job postings — skills, experience, education field,
    and target role only (name/contact info add no retrieval value).
    """
    parts = []

    if profile.target_role:
        parts.append(f"Target role: {profile.target_role}")

    if profile.skills:
        parts.append(f"Skills: {', '.join(profile.skills)}")

    if profile.experience:
        exp_lines = []
        for exp in profile.experience:
            desc = f" — {exp.description}" if exp.description else ""
            exp_lines.append(f"{exp.title} at {exp.company}{desc}")
        parts.append("Experience: " + "; ".join(exp_lines))

    if profile.education:
        edu_lines = [f"{edu.degree}" for edu in profile.education]
        parts.append("Education: " + "; ".join(edu_lines))

    if profile.summary:
        parts.append(f"Summary: {profile.summary}")

    return "\n".join(parts)