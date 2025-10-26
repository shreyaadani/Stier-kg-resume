# 🕸️ Resume / Research Knowledge Graph (Local • No API Keys)

Turn TXT/PDF resumes or paper abstracts into an **interactive knowledge graph**.

- **NLP**: spaCy NER + PhraseMatcher (skills) + regex (emails/URLs)
- **Relations**: sentence co-occurrence → `worked_at`, `uses`, `based_in`, `at`
- **Graph**: NetworkX → PyVis (interactive) embedded in Streamlit
- **Privacy**: 100% local, no external APIs/keys

## ✨ Features

- Upload **multiple TXT/PDFs** and preview extracted text
- Entity groups: **PEOPLE, ORGS, PLACES, DATES, EMAILS, URLS, SKILLS, PROJECTS**
- Relations table + **interactive graph**
- **JSON export** of nodes and edges
- Custom **skills dictionary** from sidebar

## 🧱 Stack

Python, Streamlit, spaCy (`en_core_web_sm`), NetworkX, PyVis, pypdf, pdfminer.six

## 🚀 Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -U -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```
