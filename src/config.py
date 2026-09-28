import os
from dotenv import load_dotenv
from pathlib import Path

# Project root = the folder that contains src/ (one level above this file's folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()

# --- API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Models ---
GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"   
GEMINI_PRO_MODEL = "gemini-3.5-flash"      
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JOBS_DIR = DATA_DIR / "jobs"
CAREER_NOTES_DIR = DATA_DIR / "career_notes"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")