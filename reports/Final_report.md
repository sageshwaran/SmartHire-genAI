# SmartHire GenAI — Final Project Report

## 1. Overview

SmartHire GenAI is a deployed career portal that parses an uploaded resume into
a structured profile, matches it against a job corpus using semantic search,
generates targeted CV improvement suggestions, and answers career questions
through a retrieval-augmented generation (RAG) chatbot. The project covers the
full pipeline required by the brief: structured LLM output, embeddings and
vector search, prompt engineering, RAG orchestration with LangChain, guardrails,
evaluation, and deployment.

Live app: resume-matching-and-career-mentor.streamlit.app
Repository: github.com/\sageshwaran/smarthire-genai

## 2. Design Choices

### 2.1 Single-provider strategy: Gemini only

The brief allows any LLM provider. OpenAI was dropped early in favor of Gemini
alone, for chat generation and embeddings both. This simplified configuration
(one API key, one SDK) and kept the project within a free tier for the full
build. The trade-off was exposure to Gemini's own model lifecycle — two model
names used during development (`gemini-1.5-flash`, then `text-embedding-004`)
were retired mid-project and had to be swapped for current equivalents
(`gemini-3.5-flash-lite`, `gemini-embedding-001`). This was treated as a
configuration change rather than a design flaw, since model names were never
hardcoded outside `config.py`.

### 2.2 Structured output via Pydantic schemas

Every LLM call that needs to feed downstream code (resume parsing, CV
suggestions, guardrails classification) uses a Pydantic model as the response
schema, rather than asking the model to "return JSON" in a plain prompt. This
was more work up front — Gemini's structured-output format required stripping
Pydantic-generated schema keys it doesn't support (`default`, `anyOf`,
`$ref` indirection) under the older `google-generativeai` SDK. Migrating to the
newer `google-genai` SDK partway through removed this entirely, since it
accepts Pydantic models natively. The lesson carried forward: validate LLM
output against the schema even when the API claims to guarantee it, as a
defense-in-depth step.

### 2.3 FAISS with a sampled job corpus

The job corpus (Naukri Data Science Jobs, India) has ~12,000 rows. Only a
1,500-row random sample was embedded, for two reasons: Gemini's free-tier
embedding quota is daily and limited, and a sample of this size already
produces a job corpus large enough to generate multiple plausible, competing
matches per query — the goal was demonstrating working semantic search, not
exhaustively indexing the dataset. A checkpointing mechanism was added to the
embedding build step after an initial run was interrupted by the daily quota,
so partial progress is never lost on a subsequent run.

### 2.4 Two-layer guardrails, not one

Guardrails combine a fast, free, rule-based layer (empty/oversized input,
regex-based prompt-injection detection) with a second LLM-based topic
classifier for subtler off-topic cases the rules can't catch. The rule-based
layer runs first and can short-circuit before any API call is made. This
two-layer design was applied to all three LLM-calling entry points — resume
parsing, CV suggestions, and the mentor chat — after an initial implementation
only covered the mentor chat, which was a real gap caught during a deliberate
requirements review against the original brief.

### 2.5 RAG grounding with an explicit refusal instruction

The AI Career Mentor retrieves from two separate sources per query — a small,
hand-written career notes knowledge base and the same job corpus used for
matching — then generates an answer constrained to that retrieved context. The
system prompt explicitly instructs the model to say it doesn't have enough
information rather than guess, which was verified directly in the evaluation
(see below) rather than assumed to work from the prompt wording alone.

### 2.6 Deployment: prebuilt indexes committed, not rebuilt on deploy

The FAISS indexes (job corpus and career notes) are committed to the
repository rather than rebuilt at app startup. This was a deliberate
trade-off: rebuilding from the raw CSV on every cold start would be slow, and
more importantly would burn Gemini's daily embedding quota on every deploy or
container restart, risking the live app going down shortly after launch. The
job dataset CSV itself remains out of the repository (`.gitignore`), since only
the resulting vectors are actually needed at runtime.

## 3. What Worked

**Structured output discipline** made every downstream module (search,
  suggestions, guardrails) composable, since each function could rely on a
  validated Python object rather than parsing free text.
 **Retrieval relevance was consistently strong** (96% hit rate across five
  test queries covering different target roles — see `answer_quality.md`),
  even on a sampled, India-only, data-science-skewed corpus.
**The hallucination check passed 4/4** in formal testing — the mentor
  correctly declined to answer fabricated-sounding but unanswerable questions
  (an exact salary figure, "which company is hiring the most this week")
  rather than inventing a plausible-sounding response.
**Reusing one schema-cleaning utility across three modules** (resume parser,
  guardrails classifier) under the older SDK, and reusing one guardrails
  rule-check function across all three entry points after the SDK migration,
  kept the codebase from accumulating duplicated logic as features were added.
**The documented prompt comparison** (Phase 7) showed a concrete, non-obvious
  finding: the value of the structured-output prompt over the free-text
  baseline was less about raw suggestion quality — both were reasonable — and
  more about the structured version being directly usable by the application
  without text parsing.

## 4. Limitations

**Corpus scope**: 1,500 sampled rows from a single, India-only,
  data-science-focused dataset. Match quality for resumes aimed at other
  fields, or other regions, would likely be weaker and was not tested.
**Project vs. experience extraction**: the resume parser correctly separates
  formal job `experience` from `projects`, but the job-search query currently
  only draws on skills, education, and summary — not project descriptions
  directly. For student resumes where projects carry most of the signal, this
  likely understates some candidates' fit.
**Grounding variance**: a specific grounding error (the CV suggestion
  generator once flagged "Gemini" as a missing skill despite it appearing in
  the candidate's own project description) was observed in one test run and
  did not reproduce in a later evaluation run with the same inputs. This
  points to run-to-run inconsistency in the model's grounding behavior that a
  single before/after prompt comparison cannot fully characterize.
 **Guardrail coverage**: the prompt-injection detector is a small,
  hand-written regex list. It catches the common phrasing tested during
  development but is not an exhaustive defense against more sophisticated or
  non-English injection attempts.
**Quota-bound deployment**: the live app runs on a free-tier Gemini API key
  with a daily request quota. Under sustained or heavy use, features can
  become temporarily unavailable until the quota resets. This is surfaced to
  users as a plain-language message rather than a broken page, but it remains
  a real availability constraint not present in a production system with a
  paid tier or multi-key fallback.