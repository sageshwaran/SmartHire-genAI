from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL, CAREER_NOTES_DIR, VECTORSTORE_DIR
from src.search.job_search import search_jobs

CAREER_NOTES_INDEX_DIR = VECTORSTORE_DIR / "career_notes_index"

_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


def build_career_notes_index():
    """Load career notes .txt files, chunk them, embed, and save a FAISS vectorstore."""
    notes_dir = Path(CAREER_NOTES_DIR)
    documents = []
    for file_path in notes_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": file_path.name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, _embeddings)

    CAREER_NOTES_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(CAREER_NOTES_INDEX_DIR))

    print(f"Career notes index built: {len(chunks)} chunks from {len(documents)} documents")
    return vectorstore


def load_career_notes_index():
    return FAISS.load_local(
        str(CAREER_NOTES_INDEX_DIR),
        _embeddings,
        allow_dangerous_deserialization=True,  # safe here: we generated this file ourselves
    )


def _format_job_context(job_results) -> str:
    lines = []
    for job in job_results:
        snippet = job["skills_description"][:200]
        lines.append(f"- {job['job_role']} at {job['company']} ({job['location']}): {snippet}")
    return "\n".join(lines)


MENTOR_SYSTEM_PROMPT = """You are the AI Career Mentor for SmartHire GenAI, a career guidance \
assistant for students and early-career job seekers.

Rules:
- Answer ONLY using the CONTEXT provided below (career notes and job market data). \
Do not use outside knowledge.
- If the context does not contain enough information to answer confidently, say \
"I don't have enough information to answer that confidently" instead of guessing.
- Stay focused on career, job search, and skill development topics. Politely decline \
unrelated questions.
- Be specific and practical, not generic."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", MENTOR_SYSTEM_PROMPT),
    ("human",
     "CONTEXT - Career notes:\n{notes_context}\n\n"
     "CONTEXT - Related job postings:\n{jobs_context}\n\n"
     "Question: {question}"),
])

_llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, google_api_key=GOOGLE_API_KEY)

_chain = _prompt | _llm | StrOutputParser()


def ask_mentor(question: str) -> str:
    """
    Answer a career question using RAG: retrieve relevant career notes + relevant
    job postings, then generate a grounded answer via a LangChain LCEL pipeline.
    """
    notes_store = load_career_notes_index()
    note_docs = notes_store.similarity_search(question, k=3)
    notes_context = "\n\n".join(
        f"[{d.metadata.get('source', 'note')}] {d.page_content}" for d in note_docs
    )

    job_results = search_jobs(question, top_n=3)
    jobs_context = _format_job_context(job_results)

    return _chain.invoke({
        "notes_context": notes_context,
        "jobs_context": jobs_context,
        "question": question,
    })