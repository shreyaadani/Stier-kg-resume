import io
import streamlit as st

# PDF extractors
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text

st.set_page_config(page_title="Resume / Research KG", layout="wide")
st.title("🕸️ Resume / Research Knowledge Graph (MVP)")

st.markdown(
    "Upload **TXT or PDF** files. We’ll extract text (pypdf → pdfminer fallback) and show a preview."
)

# ---------- helpers ----------
def read_pdf(file_bytes: bytes) -> str:
    """Try pypdf first (fast), then pdfminer (more robust)."""
    # Attempt 1: pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    # Attempt 2: pdfminer
    try:
        text = pdfminer_extract_text(io.BytesIO(file_bytes)) or ""
        return text.strip()
    except Exception:
        return ""

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()

def extract_text_from_upload(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return read_pdf(data)
    return read_txt(data)

# ---------- UI ----------
uploaded_files = st.file_uploader(
    "Upload sample TXT or PDF files (multiple allowed)",
    type=["txt", "pdf"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("No files yet — try uploading a TXT or PDF.")
else:
    # Read all files and show previews
    for uf in uploaded_files:
        with st.expander(f"📄 {uf.name}", expanded=False):
            try:
                text = extract_text_from_upload(uf)
                if text:
                    st.success("Text extracted ✅ (showing first ~1200 chars)")
                    st.code(text[:1200] + ("..." if len(text) > 1200 else ""), language="markdown")
                    st.caption(f"Characters extracted: {len(text)}")
                else:
                    st.warning("Could not extract any text from this file.")
            except Exception as e:
                st.error(f"Extraction error: {e}")
