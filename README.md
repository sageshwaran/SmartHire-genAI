# SmartHire GenAI

An end-to-end Generative AI career portal. Upload a resume and get a structured
candidate profile, semantically matched jobs from a real job dataset, AI-generated
CV improvement suggestions for a target role, and an AI Career Mentor that answers
career questions using retrieval-augmented generation (RAG) — grounded strictly in
a defined knowledge base, not open-ended generation.

Built as a Generative AI capstone project, covering prompt engineering, structured
LLM output, embeddings, vector search, RAG, LangChain orchestration, guardrails,
evaluation, and deployment.

## Features

- **Resume Parser** — extracts structured data (name, skills, experience, education,
  target role) from an uploaded PDF/DOCX resume using Gemini's structured output.
- **Semantic Job Search** — embeds a job postings corpus into a FAISS vector index
  and matches candidates to jobs by meaning, not keyword overlap.
- **CV Improvement Generator** — compares a resume against a target job description
  and produces specific, grounded suggestions: missing skills, weak bullet points,
  and a rewritten summary.
- **AI Career Mentor** — a RAG chatbot (LangChain + Gemini) that answers career
  questions using a curated set of career notes and live job-market data. Refuses
  to answer when the information isn't in its knowledge base.
- **Guardrails** — a two-layer validation system (rule-based pattern checks + an
  LLM-based topic classifier) that rejects empty, oversized, off-topic, or
  prompt-injection input before it reaches the main LLM calls.

## Tech Stack

- **LLM / Embeddings**: Google Gemini (`google-genai` SDK) — `gemini-3.5-flash-lite`
  for generation, `gemini-embedding-001` for embeddings
- **Orchestration**: LangChain (LCEL pipelines, RAG retrieval)
- **Vector Search**: FAISS (`faiss-cpu`)
- **Structured Output**: Pydantic schemas, enforced via Gemini's native schema support
- **UI**: Streamlit
- **Document Parsing**: pypdf, python-docx

## Project Structure
smarthire-genai/
├── data/
│ ├── jobs/ # Job postings dataset (CSV, not committed)
│ ├── resumes/ # Sample resumes for testing
│ └── career_notes/ # Career guidance documents the mentor retrieves from
├── vectorstore/ # Saved FAISS indexes (not committed)
├── src/
│ ├── config.py
│ ├── parsing/ # Resume loading + structured parsing
│ ├── search/ # Embeddings + FAISS job search
│ ├── generate/ # Prompt library + CV suggestion generator
│ ├── mentor/ # RAG career mentor (LangChain)
│ ├── safety/ # Guardrails layer
│ └── evaluate.py # Evaluation script
├── app/
│ └── streamlit_app.py # Streamlit portal
└── reports/
└── answer_quality.md # Evaluation report


## Setup

1. Clone the repository
```bash
   git clone https://github.com/sageshwaran/smarthire-genai.git
   cd smarthire-genai
```

2. Create and activate a virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Set up environment variables
```bash
   cp .env.example .env
```
   Add your Gemini API key to `.env`:
   GOOGLE_API_KEY = your_key_here


5. Add a job dataset
   Download a job postings dataset (e.g. [Naukri Data Science Jobs — Kaggle](https://www.kaggle.com/datasets/anandhuh/data-science-jobs-in-india))
   and place the CSV at `data/jobs/naukri_data_science_jobs_india.csv`.

6. Build the job search index
```python
   from src.search.job_search import build_job_index
   build_job_index("data/jobs/naukri_data_science_jobs_india.csv", sample_size=1500)
```
   This embeds the job corpus and saves a FAISS index to `vectorstore/`. One-time
   step; re-run only if the dataset changes.

7. Build the career notes index
```python
   from src.mentor.rag_chain import build_career_notes_index
   build_career_notes_index()
```

8. Run the app
```bash
   streamlit run app/streamlit_app.py
```

## Evaluation

See [`reports/answer_quality.md`](reports/answer_quality.md) for the full evaluation,
covering:
- **Retrieval relevance** — manual relevance scoring across test queries (96% hit rate)
- **Answer quality** — correctness, grounding, and helpfulness scoring for mentor responses
- **Prompt comparison** — a documented before/after comparing an unstructured baseline
  prompt against a grounded, schema-enforced prompt
- **Hallucination check** — confirms the mentor declines or admits uncertainty rather
  than fabricating answers (4/4 correct refusals in testing)

## Known Limitations

- The job corpus is a sample (1,500 rows) of a single dataset (Naukri, data-science-
  focused, India only) — match quality reflects that scope.
- Resume parsing extracts formal `experience` entries separately from `projects`;
  project-based resumes (e.g. students) currently rely on skills/education/summary
  for job matching rather than project descriptions directly.
- The injection-pattern guardrail uses a small, hand-written regex list rather than
  a more exhaustive detection system.
- LLM output has observed run-to-run variance in grounding accuracy (documented in
  the evaluation report) — occasional inconsistencies are possible.

## License

This project was built as an educational capstone project.