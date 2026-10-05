import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_secret(key: str) -> str | None:
    """Read a secret from .env (local dev) or Streamlit secrets (deployed)."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


GOOGLE_API_KEY = _get_secret("GOOGLE_API_KEY")

GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"
GEMINI_PRO_MODEL = "gemini-3.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072

DATA_DIR = PROJECT_ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JOBS_DIR = DATA_DIR / "jobs"
CAREER_NOTES_DIR = DATA_DIR / "career_notes"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Set it in .env (local) or Streamlit secrets (deployed).")