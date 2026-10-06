import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import tempfile

from src.parsing.loader import load_resume_text
from src.parsing.resume_parser import parse_resume
from src.search.job_search import match_jobs_for_profile
from src.generate.cv_suggestions import generate_cv_suggestions
from src.mentor.rag_chain import ask_mentor

def friendly_error(e: Exception) -> str:
    """Translate common API failures into calm, non-technical messages for the UI."""
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return "This feature has reached its usage limit for today. Please try again later."
    if "rejected by guardrails" in msg:
        return msg.split("guardrails: ", 1)[-1] if "guardrails: " in msg else msg
    return "Something went wrong processing your request. Please try again."

st.set_page_config(page_title="SmartHire", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    :root {
        --bg: #0e1013;
        --surface: #15181d;
        --surface-raised: #1a1e24;
        --border: #262b33;
        --text-primary: #e4e6eb;
        --text-secondary: #8b909a;
        --text-tertiary: #5c6069;
        --accent: #4f7cff;
        --accent-dim: #2d3f73;
        --mono: 'JetBrains Mono', monospace;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 980px;
    }

    [data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    h1 {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
    }
    h2 {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 0;
    }
    h3 { font-size: 0.9rem; font-weight: 600; color: var(--text-primary); }

    p, label, .stMarkdown { color: var(--text-secondary); }

    .app-subtitle {
        color: var(--text-tertiary);
        font-size: 0.85rem;
        margin-top: -0.3rem;
        margin-bottom: 1.8rem;
    }

    .field-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-tertiary);
        margin-bottom: 2px;
    }
    .field-value {
        font-size: 0.92rem;
        color: var(--text-primary);
        margin-bottom: 1.1rem;
    }

    .record-card {
        background: var(--surface-raised);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .record-title { font-size: 0.92rem; font-weight: 600; color: var(--text-primary); }
    .record-meta {
        font-family: var(--mono);
        font-size: 0.75rem;
        color: var(--text-tertiary);
        margin-top: 2px;
    }
    .record-score {
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--accent);
        float: right;
    }
    .record-body {
        font-size: 0.83rem;
        color: var(--text-secondary);
        margin-top: 8px;
        line-height: 1.5;
    }

    .status-line {
        font-size: 0.82rem;
        color: var(--text-tertiary);
        border-left: 2px solid var(--border);
        padding-left: 10px;
        margin: 0.8rem 0;
    }

    .stButton button {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 5px;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.45rem 1.1rem;
        box-shadow: none !important;
    }
    .stButton button p {
        color: #ffffff !important;
    }
    .stButton button:hover {
        background-color: #3d6af0 !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--surface-raised);
        border: 1px dashed var(--border);
        border-radius: 6px;
    }

    .stTextArea textarea, .stTextInput input {
        background-color: var(--surface-raised) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stChatMessage"] {
        background: var(--surface-raised);
        border: 1px solid var(--border);
        border-radius: 6px;
    }

    hr { border-color: var(--border); }
        /* Sidebar nav links */
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        background: transparent;
        color: var(--text-secondary);
        border: none;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.87rem;
        padding: 0.5rem 0.7rem;
        margin-bottom: 2px;
        box-shadow: none;
        transition: background 0.1s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: var(--surface-raised);
        color: var(--text-primary);
    }
    [data-testid="stSidebar"] .stButton button:focus:not(:active) {
        box-shadow: none;
    }
    /* Active nav item (primary button type) */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: var(--accent-dim);
        color: var(--text-primary);
        border-left: 2px solid var(--accent);
        border-radius: 4px 0 0 4px;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background: var(--accent-dim);
    }
        .chat-row {
        display: flex;
        margin-bottom: 10px;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }

    .chat-bubble {
        max-width: 70%;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .chat-bubble.user {
        background: var(--accent-dim);
        color: var(--text-primary);
        border: 1px solid var(--accent);
        border-bottom-right-radius: 2px;
    }
    .chat-bubble.assistant {
        background: var(--surface-raised);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-bottom-left-radius: 2px;
    }
    /* Hide Streamlit GitHub/source icon */
        [data-testid="stToolbar"] {
            display: none !important;
        }

        /* Hide the top-right menu if needed */
        #MainMenu {
            visibility: hidden;
        }

        /* Hide Streamlit footer */
        footer {
            visibility: hidden;
        }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
if "profile" not in st.session_state:
    st.session_state.profile = None
if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
with st.sidebar:
   if "page" not in st.session_state:
    st.session_state.page = "Resume"

NAV_ITEMS = ["Resume", "Matched roles", "CV review", "Career mentor"]

with st.sidebar:
    st.markdown("**SmartHire**")
    st.caption("Resume analysis and job matching")
    st.divider()

    for item in NAV_ITEMS:
        is_active = st.session_state.page == item
        if st.button(item, key=f"nav_{item}", type="primary" if is_active else "secondary"):
            st.session_state.page = item
            st.rerun()

    st.divider()
    if st.session_state.profile:
        st.markdown('<div class="field-label">Loaded resume</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{st.session_state.profile.name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-line">No resume loaded</div>', unsafe_allow_html=True)

page = st.session_state.page

# ----------------------------------------------------------------------------
# Page: Resume
# ----------------------------------------------------------------------------
if page == "Resume":
    st.markdown("# Resume")
    st.markdown('<div class="app-subtitle">Upload a document to extract a structured profile.</div>', unsafe_allow_html=True)

    MAX_UPLOAD_SIZE_MB = 5

    uploaded_file = st.file_uploader("Document", type=["pdf", "docx"], label_visibility="collapsed")

    if uploaded_file is not None:
        if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            st.markdown(f'<div class="status-line">File exceeds the {MAX_UPLOAD_SIZE_MB}MB limit.</div>', unsafe_allow_html=True)
        elif st.button("Upload"):
            with st.spinner("Parsing"):
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    resume_text = load_resume_text(tmp_path)
                    st.session_state.profile = parse_resume(resume_text)
                    st.session_state.matched_jobs = None
                except Exception as e:
                    st.markdown(f'<div class="status-line">{friendly_error(e)}</div>', unsafe_allow_html=True)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    if st.session_state.profile is not None:
        p = st.session_state.profile
        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="field-label">Name</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{p.name}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="field-label">Contact</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{p.email or "—"}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="field-label">Target role</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{p.target_role or "—"}</div>', unsafe_allow_html=True)

        st.markdown('<div class="field-label">Skills</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-value">{", ".join(p.skills) if p.skills else "None extracted"}</div>', unsafe_allow_html=True)

        if p.education:
            st.markdown('<div class="field-label">Education</div>', unsafe_allow_html=True)
            for edu in p.education:
                st.markdown(f'<div class="field-value">{edu.degree} — {edu.institution} ({edu.year or "n/a"})</div>', unsafe_allow_html=True)

        if p.experience:
            st.markdown('<div class="field-label">Experience</div>', unsafe_allow_html=True)
            for exp in p.experience:
                st.markdown(f'<div class="field-value">{exp.title}, {exp.company} ({exp.duration})</div>', unsafe_allow_html=True)

        if p.summary:
            st.markdown('<div class="field-label">Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{p.summary}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Page: Matched roles
# ----------------------------------------------------------------------------
elif page == "Matched roles":
    st.markdown("# Matched roles")
    if st.session_state.profile is None:
        st.markdown('<div class="status-line">No resume loaded. Parse one on the Resume page first.</div>', unsafe_allow_html=True)
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            top_n = st.slider("Results", min_value=3, max_value=10, value=5, label_visibility="collapsed")
        with col_b:
            run = st.button("Search")

        if run:
            try:
                with st.spinner("Searching"):
                    st.session_state.matched_jobs = match_jobs_for_profile(st.session_state.profile, top_n=top_n)
            except Exception as e:
                st.markdown(f'<div class="status-line">{friendly_error(e)}</div>', unsafe_allow_html=True)

        if st.session_state.matched_jobs:
            st.divider()
            for job in st.session_state.matched_jobs:
                st.markdown(f"""
                <div class="record-card">
                    <div class="record-title">{job['job_role']}</div>
                    <div class="record-meta">{job['company']} · {job['location']} · {job['experience']}</div>
                    <div class="record-body">{job['skills_description'][:200]}</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Page: CV review
# ----------------------------------------------------------------------------
elif page == "CV review":
    st.markdown("# CV review")
    if st.session_state.profile is None:
        st.markdown('<div class="status-line">No resume loaded. Parse one on the Resume page first.</div>', unsafe_allow_html=True)
    else:
        target_job_text = st.text_area("Target job description", height=130, placeholder="Paste a job description")

        if st.session_state.matched_jobs:
            options = ["Use pasted text"] + [f"{j['job_role']} — {j['company']}" for j in st.session_state.matched_jobs]
            selected = st.selectbox("Or select a matched role", options)
            if selected != "Use pasted text":
                idx = options.index(selected) - 1
                target_job_text = st.session_state.matched_jobs[idx]["skills_description"]

        if st.button("Run review"):
            if not target_job_text.strip():
                st.markdown('<div class="status-line">Provide a job description first.</div>', unsafe_allow_html=True)
            else:
                suggestions = None
                try:
                    with st.spinner("Reviewing"):
                        suggestions = generate_cv_suggestions(st.session_state.profile, target_job_text)
                except Exception as e:
                    st.markdown(f'<div class="status-line">{friendly_error(e)}</div>', unsafe_allow_html=True)
                if suggestions:
                    st.divider()

                if suggestions.missing_skills:
                    st.markdown('<div class="field-label">Missing skills</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="field-value">{", ".join(suggestions.missing_skills)}</div>', unsafe_allow_html=True)

                if suggestions.weak_bullet_points:
                    st.markdown('<div class="field-label">Weak points</div>', unsafe_allow_html=True)
                    for wbp in suggestions.weak_bullet_points:
                        st.markdown(f"""
                        <div class="record-card">
                            <div class="record-title">{wbp.original_line}</div>
                            <div class="record-body"><b>Issue:</b> {wbp.issue}<br><b>Rewrite:</b> {wbp.suggested_rewrite}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('<div class="field-label">Suggested summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{suggestions.rewritten_summary}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Page: Career mentor
# ----------------------------------------------------------------------------
elif page == "Career mentor":
    st.markdown("# Career mentor")
    st.markdown('<div class="app-subtitle">Unsupported or off-topic questions are declined.</div>', unsafe_allow_html=True)

    for role, message in st.session_state.chat_history:
        st.markdown(f"""
        <div class="chat-row {role}">
            <div class="chat-bubble {role}">{message}</div>
        </div>
        """, unsafe_allow_html=True)

    user_question = st.chat_input("Ask a question")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))

        with st.spinner("Thinking"):
            try:
                answer = ask_mentor(user_question)
            except Exception as e:
                answer = friendly_error(e)

        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()