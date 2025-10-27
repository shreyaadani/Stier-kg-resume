# nlp/entities.py
# Complete module: spaCy NER + dynamic skill discovery (no user input, no API keys)

import re
from typing import List, Dict, Set, Optional

import spacy
from spacy.matcher import PhraseMatcher

# Optional, free local embeddings for auto-skill discovery
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------
# Base seeds (small, generic)
# -------------------------
DEFAULT_SKILLS: List[str] = [
    "python","java","c#","sql","pytorch","tensorflow","opencv","transformers",
    "hugging face","sklearn","scikit-learn","xgboost","docker","kubernetes",
    "aws","gcp","azure","spark","hadoop","airflow","postgresql","mongodb",
    "langchain","streamlit","flask","fastapi","grpc","linux","git","kafka",
    "redis","elastic","neo4j","d3","react","typescript","node","shap","graphql"
]

# Minimal heading hints (kept tiny; dynamic logic does most of the work)
LIKELY_SECTION_WORDS: Set[str] = {
    "summary","experience","work experience","education","projects","skills",
    "technologies","frameworks","tools","certifications","awards","publications"
}
BULLET_PREFIXES = ("•","-","—","*","·")

# -------------------------
# Module-level singletons
# -------------------------
_nlp: Optional[spacy.Language] = None
_skill_matcher: Optional[PhraseMatcher] = None
_embedder: Optional[SentenceTransformer] = None

# -------------------------
# Loaders / builders
# -------------------------
def load_nlp() -> spacy.Language:
    """Lazily load spaCy English model once per process."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def build_skill_matcher(skills: List[str]) -> PhraseMatcher:
    """(Re)build a PhraseMatcher from provided skills (case-insensitive)."""
    global _skill_matcher
    nlp = load_nlp()
    m = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(s) for s in skills]
    m.add("SKILL", patterns)
    _skill_matcher = m
    return m

def _load_embedder() -> SentenceTransformer:
    """Small, fast, free model suitable for CPU."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

# -------------------------
# Small utilities
# -------------------------
def _strip_bullets_and_noise(s: str) -> str:
    s = s.strip()
    while s.startswith(BULLET_PREFIXES):
        s = s[1:].strip()
    s = re.sub(r"\s+", " ", s).strip("·•-—*·, ")
    return s

def _clean_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def _looks_like_section_heading(s: str) -> bool:
    t = s.lower().strip(": ")
    if t in LIKELY_SECTION_WORDS:
        return True
    # very short and ends with colon or TitleCase single token
    return (len(t) <= 24 and (s.endswith(":") or t in LIKELY_SECTION_WORDS))

# -------------------------
# Candidate mining (automatic)
# -------------------------
def _mine_candidate_terms(text: str) -> List[str]:
    """
    Pull plausible skill-like candidates without user input:
    - noun chunks, proper nouns, single nouns
    - simple bigrams/trigrams from lines
    """
    nlp = load_nlp()
    doc = nlp(text)
    cands: Set[str] = set()

    # Noun chunks / single tokens (proper nouns & nouns)
    for nc in doc.noun_chunks:
        t = nc.text.strip()
        if 2 <= len(t) <= 40:
            cands.add(t)
    for tok in doc:
        if tok.pos_ in ("PROPN", "NOUN") and tok.is_alpha and len(tok.text) > 2:
            cands.add(tok.text)

    # Simple n-grams (2–3 words)
    for line in text.splitlines():
        parts = [p.strip(",.;:()[]{} ") for p in line.split() if p.strip(",.;:()[]{} ")]
        for i in range(len(parts)):
            for k in (2, 3):
                if i + k <= len(parts):
                    ngram = " ".join(parts[i:i+k])
                    if 3 <= len(ngram) <= 40 and any(ch.isalpha() for ch in ngram):
                        cands.add(ngram)

    # Normalize
    cands = { _strip_bullets_and_noise(c).lower() for c in cands if c and not c.isnumeric() }
    cands = { c for c in cands if c }  # drop empties
    return list(cands)

def _promote_skills_via_embeddings(
    text: str,
    base_skills: List[str],
    top_k: int = 50,
    sim_thresh: float = 0.55
) -> List[str]:
    """
    Grow skills automatically using semantic similarity:
    1) mine candidate terms from the document
    2) embed seeds and candidates
    3) keep candidates whose cosine sim to ANY seed ≥ threshold
    4) return merged unique list
    """
    embedder = _load_embedder()

    seed = sorted({s.lower().strip() for s in base_skills})
    cands = _mine_candidate_terms(text)
    if not cands:
        return seed

    # Encode (normalized for cosine via dot-product)
    seed_vecs = embedder.encode(seed, normalize_embeddings=True)
    cand_vecs = embedder.encode(cands, normalize_embeddings=True)

    # Max similarity to any seed
    sims = np.matmul(cand_vecs, seed_vecs.T).max(axis=1)
    idx = np.where(sims >= sim_thresh)[0]
    if idx.size == 0:
        return seed

    # Rank by similarity (desc), pick top_k
    order = np.argsort(-sims[idx])
    promoted = [cands[i] for i in order[:top_k]]

    merged = list(dict.fromkeys([*seed, *promoted]))  # preserve order, dedupe
    return merged

# -------------------------
# Main extraction API
# -------------------------
def extract_entities_blocks(text: str, skills: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Returns a dict of lists:
      PEOPLE, ORGS, PLACES, DATES, EMAILS, URLS, SKILLS, PROJECTS

    Pipeline:
      1) auto-expand skills from the document via embeddings
      2) build PhraseMatcher on-the-fly
      3) run spaCy NER + regex + heuristics
      4) dynamic re-assignment: if a token is in auto-skills → SKILLS wins
    """
    nlp = load_nlp()

    # 1) Auto-expand skills (no user input)
    base_skills = skills if skills else DEFAULT_SKILLS
    auto_skills = _promote_skills_via_embeddings(text, base_skills)

    # 2) Build matcher from dynamic skills
    matcher = build_skill_matcher(auto_skills)

    # 3) Core extraction
    doc = nlp(text)

    people = list({e.text for e in doc.ents if e.label_ == "PERSON"})
    orgs   = list({e.text for e in doc.ents if e.label_ == "ORG"})
    places = list({e.text for e in doc.ents if e.label_ in ("GPE","LOC")})
    dates  = list({e.text for e in doc.ents if e.label_ == "DATE"})

    emails = list({m.group(0) for m in re.finditer(r"[\w\.-]+@[\w\.-]+\.\w+", text, re.I)})
    urls   = list({m.group(0) for m in re.finditer(r"(https?://[^\s)]+)", text, re.I)})

    # Skills via phrase matcher across sentences (tokenization-aware)
    skills_found: Set[str] = set()
    for sent in doc.sents:
        span_doc = nlp(sent.text)  # local doc for indices
        for _, a, b in matcher(span_doc):
            skills_found.add(span_doc[a:b].text.lower())

    # Projects (simple heuristic): lines with build/develop/implement
    projects = set()
    for line in text.splitlines():
        l = _strip_bullets_and_noise(line)
        if re.search(r"\b(project|built|developed|created|implemented|designed|led)\b", l, re.I):
            projects.add(_clean_ws(l[:140]))

    # 4) Dynamic re-assignment: if string ∈ auto_skills → SKILLS wins
    auto_skill_set = set(auto_skills)

    def _norm_list(xs: List[str]) -> List[str]:
        ys: List[str] = []
        seen: Set[str] = set()
        for v in xs:
            v0 = _strip_bullets_and_noise(v)
            if not v0:
                continue
            # drop obvious section headings
            if _looks_like_section_heading(v0):
                continue
            key = v0.lower()
            if key in seen:
                continue
            seen.add(key)
            ys.append(v0)
        return ys

    people  = _norm_list(people)
    orgs    = _norm_list(orgs)
    places  = _norm_list(places)
    dates   = _norm_list(dates)
    emails  = _norm_list(emails)
    urls    = _norm_list(urls)
    projects= _norm_list(list(projects))

    # remove any item that is clearly a discovered skill
    def _filter_by_autoskills(items: List[str]) -> List[str]:
        return [x for x in items if x.lower() not in auto_skill_set]

    people  = _filter_by_autoskills(people)
    orgs    = _filter_by_autoskills(orgs)
    places  = _filter_by_autoskills(places)

    # merge skills: phrase hits + auto-discovered list (keep lowercased)
    skills_out = sorted(list(dict.fromkeys([*skills_found, *auto_skills])))

    return {
        "PEOPLE": people,
        "ORGS": orgs,
        "PLACES": places,
        "DATES": dates,
        "EMAILS": emails,
        "URLS": urls,
        "SKILLS": skills_out,
        "PROJECTS": sorted(list(projects)),
    }
