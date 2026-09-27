import time
import random
from typing import List
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from src.config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL

genai.configure(api_key=GOOGLE_API_KEY)

# Free tier is limited (observed: ~100 embed requests/minute).
# Keep batches small and pace requests to stay under that ceiling.
BATCH_SIZE = 10
BASE_DELAY_SECONDS = 3.0
MAX_RETRIES = 5


def _embed_with_retry(batch: List[str], task_type: str):
    """Call Gemini's embed_content with exponential backoff on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            return genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=batch,
                task_type=task_type,
            )
        except ResourceExhausted as e:
            if attempt == MAX_RETRIES - 1:
                raise  # out of retries, let it fail loudly
            # Exponential backoff with a little jitter: 4s, 8s, 16s, 32s (+ 0-1s random)
            wait_time = (2 ** (attempt + 2)) + random.uniform(0, 1)
            print(f"Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(wait_time)


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Embed a list of texts using Gemini's embedding model, in small paced batches
    with automatic retry on rate limits.
    """
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        result = _embed_with_retry(batch, task_type)
        all_embeddings.extend(result["embedding"])

        time.sleep(BASE_DELAY_SECONDS)

        print(f"Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} texts...")

    return all_embeddings


def embed_single_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    """Embed a single piece of text (e.g. a candidate profile at query time)."""
    result = _embed_with_retry([text], task_type)
    return result["embedding"][0]