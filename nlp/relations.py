from typing import Dict, List, Tuple
from .entities import load_nlp

def infer_relationships(text: str, ents: Dict[str, List[str]]) -> List[Tuple[str,str,str]]:
    """Edges (src, relation, dst) via sentence co-occurrence heuristics."""
    nlp = load_nlp()
    doc = nlp(text)
    sentences = [s.text.lower() for s in doc.sents]

    def co_occurs(a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return any(a in s and b in s for s in sentences)

    edges = []
    for p in ents.get("PEOPLE", []):
        for o in ents.get("ORGS", []):
            if co_occurs(p, o): edges.append((p, "worked_at", o))
    for pr in ents.get("PROJECTS", []):
        for sk in ents.get("SKILLS", []):
            if co_occurs(pr, sk): edges.append((pr, "uses", sk))
    for p in ents.get("PEOPLE", []):
        for pl in ents.get("PLACES", []):
            if co_occurs(p, pl): edges.append((p, "based_in", pl))
    for pr in ents.get("PROJECTS", []):
        for o in ents.get("ORGS", []):
            if co_occurs(pr, o): edges.append((pr, "at", o))

    return list({(s,r,d) for (s,r,d) in edges})

# --- Evidence helper: show sentences that triggered each edge ---
from typing import Dict, List, Tuple
from .entities import load_nlp

def collect_edge_evidence(text: str, edges: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    For each (src, relation, dst), find sentences containing both endpoints.
    Returns list of dicts: {source, relation, target, count, examples}
    """
    nlp = load_nlp()
    sents = [s.text for s in nlp(text).sents]

    def in_same_sent(a: str, b: str, sent: str) -> bool:
        a_l, b_l, s_l = a.lower(), b.lower(), sent.lower()
        return (a_l in s_l) and (b_l in s_l)

    results = []
    for s, r, d in edges:
        hits = [sent for sent in sents if in_same_sent(s, d, sent)]
        # keep at most 2 short examples
        examples = [h.strip()[:220] for h in hits[:2]]
        results.append({
            "source": s,
            "relation": r,
            "target": d,
            "count": len(hits),
            "examples": examples
        })
    return results
