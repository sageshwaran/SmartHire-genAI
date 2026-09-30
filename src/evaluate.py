"""
Evaluation script for SmartHire GenAI. Run each section independently and
record results in reports/answer_quality.md.
"""

from src.parsing.loader import load_resume_text
from src.search.job_search import search_jobs
from src.mentor.rag_chain import ask_mentor
from src.generate.cv_suggestions import generate_cv_suggestions
from src.generate.prompts import cv_improvement_prompt_v1, cv_improvement_prompt_v2
from src.parsing.resume_parser import ResumeProfile, parse_resume


# ============================================================
# PART A: Retrieval Relevance
# ============================================================

def evaluate_retrieval_relevance():
    """
    For each test query, print the top-5 matches so you can manually judge
    relevance (yes/no) and compute a hit rate by hand — this keeps the
    judgment honest and human, which is appropriate at this project's scale.
    """
    test_queries = [
        "Python developer skilled in machine learning, NLP, and data analysis",
        "Entry-level data analyst with SQL and Excel skills",
        "Backend software engineer with Django and REST APIs",
        "Data scientist with deep learning and computer vision experience",
        "Business analyst with Power BI and stakeholder communication skills",
    ]

    for query in test_queries:
        print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}")
        results = search_jobs(query, top_n=5)
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['score']:.3f}] {r['job_role']} at {r['company']}")

# ============================================================
# PART B: Answer Quality (Mentor)
# ============================================================

def evaluate_answer_quality():
    """
    Ask the mentor a set of realistic career questions. For each, manually
    judge: correctness, grounding (does it stick to the documents?), and
    helpfulness (1-5 each), recorded in the report.
    """
    test_questions = [
        "How do I switch to a Data Analyst role from a general CS background?",
        "What skills do I need to become a GenAI Engineer?",
        "What should I focus on to prepare for a Software Engineer interview?",
        "How should I structure my resume if I don't have work experience?",
    ]

    for question in test_questions:
        print(f"\n{'='*70}\nQ: {question}\n{'='*70}")
        answer = ask_mentor(question)
        print(answer)

# ============================================================
# PART C: Prompt Comparison (Before/After)
# ============================================================

def evaluate_prompt_comparison(profile: ResumeProfile):
    """
    Compare v1 (baseline) vs v2 (grounded/structured) CV improvement prompts
    on the same input, to demonstrate the concrete effect of prompt engineering.
    """
    from src.generate.cv_suggestions import _resume_to_summary_text
    from google import genai
    from google.genai import types
    from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL

    client = genai.Client(api_key=GOOGLE_API_KEY)
    resume_summary = _resume_to_summary_text(profile)

    target_job_description = """
    We are looking for a GenAI Engineer Intern with experience in:
    Python, LLM APIs (OpenAI, Gemini), LangChain, RAG pipelines, FAISS/Pinecone, Docker.
    """

    print("=" * 70)
    print("PROMPT V1 (baseline, unstructured):")
    print("=" * 70)
    response_v1 = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=cv_improvement_prompt_v1(resume_summary, target_job_description),
    )
    print(response_v1.text)

    print("\n" + "=" * 70)
    print("PROMPT V2 (grounded, structured output):")
    print("=" * 70)
    suggestions = generate_cv_suggestions(profile, target_job_description)
    print(suggestions.model_dump_json(indent=2))
    
# ============================================================
# PART D: Hallucination Check
# ============================================================

def evaluate_hallucination_check():
    """
    Confirm the mentor refuses or says it doesn't know when the answer isn't
    in the documents, rather than fabricating an answer.
    """
    test_cases = [
        "What's the exact average salary for a Data Scientist in Chennai in 2026?",
        "What's the best pizza topping?",
        "Which specific company is hiring the most GenAI engineers this week?",
        "What's your favorite programming language?",
    ]

    for question in test_cases:
        print(f"\nQ: {question}")
        print(f"A: {ask_mentor(question)}")


if __name__ == "__main__":
    from pathlib import Path
    import sys

    project_root = next(
        (path for path in [Path.cwd(), *Path.cwd().parents] if (path / "src").is_dir()),
        Path.cwd().parent,
    )
    sys.path.insert(0, str(project_root))
    resume_path = project_root / "data" / "resumes" / "sample_resume.pdf"
    resume_text = load_resume_text(str(resume_path))
    profile = parse_resume(resume_text)
    evaluate_prompt_comparison(profile)
    evaluate_hallucination_check()