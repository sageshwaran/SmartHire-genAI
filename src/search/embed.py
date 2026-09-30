import time
import random
from typing import List
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from src.config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL

client = genai.Client(api_key=GOOGLE_API_KEY)

BATCH_SIZE = 10
BASE_DELAY_SECONDS = 3.0
MAX_RETRIES = 5


def _is_rate_limit_error(e: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)


def _embed_with_retry(batch: List[str], task_type: str):
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except ClientError as e:
            if not _is_rate_limit_error(e) or attempt == MAX_RETRIES - 1:
                raise
            wait_time = (2 ** (attempt + 2)) + random.uniform(0, 1)
            print(f"Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(wait_time)


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        result = _embed_with_retry(batch, task_type)
        all_embeddings.extend([e.values for e in result.embeddings])

        time.sleep(BASE_DELAY_SECONDS)
        print(f"Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} texts...")

    return all_embeddings


def embed_single_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    result = _embed_with_retry([text], task_type)
    return result.embeddings[0].values