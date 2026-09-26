from pypdf import PdfReader
from docx import Document


def load_resume_text(file_path: str) -> str:
    """
    Load a resume file (PDF or DOCX) and return its raw text content.
    """
    if file_path.lower().endswith(".pdf"):
        return _load_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return _load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Use PDF or DOCX.")


def _load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def _load_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    return text.strip()