import io
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text

def read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass
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
    return read_pdf(data) if name.endswith(".pdf") else read_txt(data)
