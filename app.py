# app.py — minimal, modular wiring (uploads → extract text → show entities)
import streamlit as st


# our modules (from Step 6R)
from nlp.extract_text import extract_text_from_upload
from nlp.entities import extract_entities_blocks, DEFAULT_SKILLS, build_skill_matcher
from nlp.relations import infer_relationships
from graph.build import build_graph
from graph.visualize import graph_to_html
from nlp.relations import infer_relationships, collect_edge_evidence




st.set_page_config(page_title="Resume / Research KG", layout="wide")
st.title("🕸️ Resume / Research Knowledge Graph (Modular MVP)")

st.markdown(
    "Upload **TXT/PDF** files. We extract text and run **spaCy NER + Skills PhraseMatcher**."
)

# ---- Sidebar: skills dictionary (optional extend) ----
with st.sidebar:
    st.header("Skills Dictionary")
    skills_text = st.text_area(
        "Extend skills (comma-separated)",
        value=", ".join(DEFAULT_SKILLS),
        height=120,
    )
    custom_skills = [s.strip() for s in skills_text.split(",") if s.strip()]
    if st.button("Reload Skill Matcher"):
        build_skill_matcher(custom_skills)  # initializes internal matcher
        st.success("Skills matcher updated.")

# ---- File upload ----
uploaded_files = st.file_uploader(
    "Upload TXT/PDF (multiple allowed)", type=["txt", "pdf"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("No files yet — try uploading a TXT or PDF.")
    st.stop()

# ---- Preview per file ----
texts = []
for uf in uploaded_files:
    with st.expander(f"📄 {uf.name}", expanded=False):
        try:
            uf.seek(0)
            text = extract_text_from_upload(uf)
            texts.append(text)
            if text:
                st.success("Text extracted ✅ (showing first ~1000 chars)")
                st.code(text[:1000] + ("..." if len(text) > 1000 else ""), language="markdown")
                st.caption(f"Characters extracted: {len(text)}")
            else:
                st.warning("No text extracted.")
        except Exception as e:
            st.error(f"Extraction error: {e}")

# ---- Merge and run NER/skills once ----
merged = "\n\n".join(t for t in texts if t).strip()
if not merged:
    st.warning("No text available to analyze.")
    st.stop()

st.subheader("🔎 Extracted Entities")
ents = extract_entities_blocks(merged)

# ---- Toggles & compact display ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    keep_people = st.checkbox("People", True)
    keep_orgs = st.checkbox("Organizations", True)
with c2:
    keep_places = st.checkbox("Places", True)
    keep_dates = st.checkbox("Dates", False)
with c3:
    keep_emails = st.checkbox("Emails", True)
    keep_urls = st.checkbox("URLs", False)
with c4:
    keep_skills = st.checkbox("Skills", True)
    keep_projects = st.checkbox("Projects", True)

mask = {
    "PEOPLE": keep_people, "ORGS": keep_orgs, "PLACES": keep_places, "DATES": keep_dates,
    "EMAILS": keep_emails, "URLS": keep_urls, "SKILLS": keep_skills, "PROJECTS": keep_projects,
}
ents = {k: v for k, v in ents.items() if mask.get(k, True)}

cols = st.columns(4)
groups = list(ents.keys())
for i, g in enumerate(groups):
    with cols[i % 4]:
        st.write(f"**{g}** ({len(ents[g])})")
        st.caption(", ".join(sorted(ents[g])[:40]) or "—")
# ======================================================
# 🔗 Relationship Inference Section
# ======================================================
st.subheader("🔗 Inferred Relationships")

edges = infer_relationships(merged, ents)
st.write(f"Edges inferred: **{len(edges)}**")

if edges:
    st.dataframe(
        [{"source": s, "relation": r, "target": d} for (s, r, d) in sorted(edges)],
        use_container_width=True,
    )

    # 🧪 Edge Evidence panel
    with st.expander("🧪 Edge Evidence (sentences that triggered each edge)", expanded=False):
        show_evidence = st.checkbox(
            "Show evidence",
            value=False,
            help="Display 1–2 example sentences and counts per edge"
        )
        if show_evidence:
            evid = collect_edge_evidence(merged, edges)
            for row in evid:
                row["examples"] = " • ".join(row["examples"]) if row["examples"] else "—"
            st.dataframe(evid, use_container_width=True)

else:
    st.info("No relations inferred yet. Try adding more files or enabling more entity types.")


# ======================================================
# 🕸️  Interactive Knowledge Graph Section
# ======================================================
st.subheader("🕸️ Interactive Knowledge Graph")

# 1️⃣ Build the NetworkX graph
G = build_graph(ents, edges)

# 2️⃣ Render to interactive HTML and embed
try:
    html = graph_to_html(G, height="760px", physics=True)
    st.components.v1.html(html, height=780, scrolling=True)
except Exception as e:
    st.error(f"Graph render failed: {e}")
    st.stop()

# 3️⃣ Optional: JSON export (recruiter-friendly data format)
import json, base64
st.divider()
st.caption("⬇️ Download graph as JSON")

try:
    nodes = [{"id": n, "group": G.nodes[n].get("group")} for n in G.nodes()]
    edges_json = [
        {"source": s, "relation": data.get("label", ""), "target": t}
        for s, t, data in G.edges(data=True)
    ]
    payload = json.dumps({"nodes": nodes, "edges": edges_json}, indent=2)
    b64 = base64.b64encode(payload.encode()).decode()
    st.markdown(
        f'<a download="knowledge_graph.json" '
        f'href="data:application/json;base64,{b64}">'
        f'Download graph JSON</a>',
        unsafe_allow_html=True,
    )
except Exception as e:
    st.error(f"JSON export failed: {e}")

