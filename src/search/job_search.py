import pickle
from pathlib import Path
import numpy as np
import faiss
import pandas as pd

from src.config import VECTORSTORE_DIR
from src.search.embed import embed_texts, embed_single_text
from src.parsing.resume_parser import ResumeProfile
from src.search.profile_query import profile_to_search_text



INDEX_PATH = Path(VECTORSTORE_DIR) / "jobs.index"
METADATA_PATH = Path(VECTORSTORE_DIR) / "jobs_metadata.pkl"
CHECKPOINT_PATH = Path(VECTORSTORE_DIR) / "embeddings_checkpoint.pkl"


def _job_to_text(row: pd.Series) -> str:
    return (
        f"Job Title: {row['Job_Role']}\n"
        f"Company: {row['Company']}\n"
        f"Experience Required: {row['Job Experience']}\n"
        f"Skills and Description: {row['Skills/Description']}"
    )


def build_job_index(csv_path: str, sample_size: int = None, checkpoint_every: int = 50):
    """
    Build a FAISS index from the job dataset. Saves progress every
    `checkpoint_every` items so a daily-quota cutoff (or crash) doesn't
    lose already-computed embeddings — re-running resumes automatically.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Skills/Description"]).reset_index(drop=True)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    job_texts = [_job_to_text(row) for _, row in df.iterrows()]

    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint = pickle.load(f)
        embeddings = checkpoint["embeddings"]
        start_idx = len(embeddings)
        print(f"Resuming from checkpoint: {start_idx}/{len(job_texts)} already embedded.")
    else:
        embeddings = []
        start_idx = 0

    remaining_texts = job_texts[start_idx:]

    if not remaining_texts:
        print("All texts already embedded — proceeding to build index.")
    else:
        for i in range(0, len(remaining_texts), checkpoint_every):
            chunk = remaining_texts[i:i + checkpoint_every]
            try:
                chunk_embeddings = embed_texts(chunk, task_type="RETRIEVAL_DOCUMENT")
            except Exception as e:
                # Save whatever we have so far before propagating the error
                with open(CHECKPOINT_PATH, "wb") as f:
                    pickle.dump({"embeddings": embeddings}, f)
                print(f"Stopped at {len(embeddings)}/{len(job_texts)} — progress saved. "
                      f"Re-run this same call tomorrow (after the daily quota resets) to continue.")
                raise

            embeddings.extend(chunk_embeddings)

            Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({"embeddings": embeddings}, f)

            print(f"Checkpoint saved: {len(embeddings)}/{len(job_texts)} total embeddings.")

    embeddings_array = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings_array)

    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)

    faiss.write_index(index, str(INDEX_PATH))

    metadata = df.to_dict(orient="records")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"Index built and saved: {index.ntotal} vectors")
    return index, metadata


def load_job_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search_jobs(query_text: str, top_n: int = 5):
    index, metadata = load_job_index()

    query_embedding = embed_single_text(query_text, task_type="RETRIEVAL_QUERY")
    query_vector = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_n)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        job = metadata[idx]
        results.append({
            "score": float(score),
            "job_role": job["Job_Role"],
            "company": job["Company"],
            "location": job["Location"],
            "experience": job["Job Experience"],
            "skills_description": job["Skills/Description"],
        })

    return results

def match_jobs_for_profile(profile: ResumeProfile, top_n: int = 5):
    """
    Given a parsed resume profile, find the top_n most semantically similar jobs.
    This is the full pipeline entry point: profile -> query text -> embedding -> FAISS search.
    """
    query_text = profile_to_search_text(profile)
    return search_jobs(query_text, top_n=top_n)