# SmartHire GenAI — Evaluation Report

## 1. Retrieval Relevance

**Methodology**: 5 test queries covering different target roles (Python/ML,
Data Analyst, Backend Engineer, Data Scientist/CV, Business Analyst), top-5
results each, manually judged relevant/not relevant against a 950-job sample
of the Naukri Data Science Jobs (India) dataset.

| Query | Relevant / 5 | Notes |
|---|---|---|
| Python developer (ML, NLP, data analysis) | 5/5 | Strong topical match across all results |
| Entry-level data analyst (SQL, Excel) | 5/5 | All Data Analyst / MIS roles |
| Backend engineer (Django, REST APIs) | 4/5 | 1 result ("Software Developer - DS") data-flavored rather than pure backend |
| Data scientist (deep learning, CV) | 5/5 | Includes a Walmart and KPMG listing — strong relevance |
| Business analyst (Power BI, stakeholder comms) | 5/5 | All Business/Data Analyst roles |

**Hit rate: 24/25 (96%)**

## 2. Answer Quality (AI Career Mentor)

| Question | Correctness | Grounding | Helpfulness | Notes |
|---|---|---|---|---|
| Switch to Data Analyst from CS background | 5/5 | 5/5 | 5/5 | Blended career notes with live job-corpus data (SAS, Postgres) |
| Skills needed for GenAI Engineer | 5/5 | 5/5 | 5/5 | Directly matches career notes roadmap |
| Prepare for Software Engineer interview | 5/5 | 5/5 | 4/5 | Solid, could add more concrete example prompts |
| Structure resume with no work experience | 5/5 | 5/5 | 5/5 | Matches resume_and_interview_tips.txt closely |

## 3. Prompt Comparison (Before/After)

**v1 (baseline, unstructured)**: produced high-quality, thoughtful advice as
free-form Markdown — readable, but not machine-usable. Extracting
`missing_skills` or a clean summary for the app's UI would require fragile
text parsing.

**v2 (grounded, structured output)**: produced equivalent insight (same core
missing skills, comparable improved summary) as schema-validated JSON,
directly consumable by the Streamlit app with no parsing needed.

**Conclusion**: the key benefit of the v2 prompt + structured output design
wasn't raw content quality (both versions were good) — it was **production
usability**. This is the practical reason for using structured output in
an application pipeline, beyond just response quality.

**Additional finding**: an earlier test run (Phase 2) showed v2 incorrectly
flagging "Gemini" as a missing skill despite it appearing in the candidate's
project description. This evaluation run did not reproduce that error —
indicating an inconsistent, run-to-run grounding issue rather than a reliable
bug. Documented here as a known limitation rather than a claimed fix.

## 4. Hallucination Check

| Question | Response | Correctly refused/admitted uncertainty? |
|---|---|---|
| Exact avg. salary, Data Scientist, Chennai, 2026 | "I don't have enough information to answer that confidently." | Yes |
| Best pizza topping | Declined as off-topic | Yes |
| Which company hiring most GenAI engineers this week | "I don't have enough information to answer that confidently." | Yes |
| Favorite programming language | Declined as off-topic | Yes |

**4/4 correct refusals — no hallucinated answers.**

## 5. Limitations

- Job corpus limited to 950 sampled rows from the Naukri Data Science Jobs
  (India) dataset — biased toward data science/analyst roles, and India-only.
- Resume `experience` extraction returns empty for project-based resumes
  (student profiles with projects rather than formal jobs) — search relies
  on skills/education/summary only in that case, not project descriptions.
- The "Gemini" skill-grounding gap observed in Phase 2 was not reproduced in
  this evaluation run, suggesting inconsistency rather than a fixed or
  reliably-reproducible issue — worth monitoring with more trials.
- Injection-pattern guardrails use a small, hand-written regex list; a
  production system would want broader coverage.