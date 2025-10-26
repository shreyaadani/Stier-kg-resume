import re
import spacy
from spacy.matcher import PhraseMatcher

DEFAULT_SKILLS = [
    "python","java","c#","sql","pytorch","tensorflow","opencv","transformers",
    "hugging face","sklearn","scikit-learn","xgboost","docker","kubernetes",
    "aws","gcp","azure","spark","hadoop","airflow","postgresql","mongodb",
    "langchain","streamlit","flask","fastapi","grpc","linux","git","kafka",
    "redis","elastic","neo4j","d3","react","typescript","node","shap"
]

_nlp = None
_skill_matcher = None

def load_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def build_skill_matcher(skills=None):
    global _skill_matcher
    if skills is None:
        skills = DEFAULT_SKILLS
    nlp = load_nlp()
    m = PhraseMatcher(nlp.vocab, attr="LOWER")
    m.add("SKILL", [nlp.make_doc(s) for s in skills])
    _skill_matcher = m
    return _skill_matcher

def _clean_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def extract_entities_blocks(text: str, skills=None):
    nlp = load_nlp()
    matcher = _skill_matcher or build_skill_matcher(skills)

    doc = nlp(text)

    people = list({e.text for e in doc.ents if e.label_ == "PERSON"})
    orgs   = list({e.text for e in doc.ents if e.label_ == "ORG"})
    places = list({e.text for e in doc.ents if e.label_ in ("GPE","LOC")})
    dates  = list({e.text for e in doc.ents if e.label_ == "DATE"})

    emails = list({m.group(0) for m in re.finditer(r"[\w\.-]+@[\w\.-]+\.\w+", text, re.I)})
    urls   = list({m.group(0) for m in re.finditer(r"(https?://[^\s)]+)", text, re.I)})

    skills_found = set()
    for sent in doc.sents:
        for _, a, b in matcher(nlp(sent.text)):
            skills_found.add(sent[a:b].text.lower())
    skills_found = list(skills_found)

    projects = set()
    for line in text.splitlines():
        l = line.strip()
        if re.search(r"\b(project|built|developed|created|implemented)\b", l, re.I):
            projects.add(_clean_ws(l[:140]))
    projects = list(projects)

    return {
        "PEOPLE": people, "ORGS": orgs, "PLACES": places, "DATES": dates,
        "EMAILS": emails, "URLS": urls, "SKILLS": skills_found, "PROJECTS": projects
    }
