import os
from dotenv import load_dotenv

load_dotenv()

# --- API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Models ---
GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"   
GEMINI_PRO_MODEL = "gemini-3.5-flash"      
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"  

# --- Paths ---
DATA_DIR = "data"
RESUMES_DIR = f"{DATA_DIR}/resumes"
JOBS_DIR = f"{DATA_DIR}/jobs"
CAREER_NOTES_DIR = f"{DATA_DIR}/career_notes"
VECTORSTORE_DIR = "vectorstore"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")