from pyvis.network import Network
import tempfile
import os

COLOR_MAP = {
    "PEOPLE": "#4C78A8", "ORGS": "#F58518", "PLACES": "#54A24B", "DATES": "#B279A2",
    "EMAILS": "#E45756", "URLS": "#72B7B2", "SKILLS": "#FF9DA6", "PROJECTS": "#9C755F",
    "DEFAULT": "#999999",
}

def graph_to_html(G, height="720px", physics=True) -> str:
    """
    Build a PyVis Network from a NetworkX graph and return clean HTML string.
    Avoids file I/O to prevent JSON decode issues on Windows.
    """
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.toggle_physics(physics)

    # Add nodes and edges
    for node, data in G.nodes(data=True):
        group = data.get("group", "DEFAULT")
        color = COLOR_MAP.get(group, COLOR_MAP["DEFAULT"])
        net.add_node(node, label=node, title=data.get("title", node), color=color)

    for s, t, d in G.edges(data=True):
        label = d.get("label", "")
        net.add_edge(s, t, label=label, title=label, arrows="to")

    net.set_options("""
{
  "nodes": { "shape": "dot", "scaling": { "min": 5, "max": 25 } },
  "edges": { "smooth": true, "font": { "size": 12, "align": "top" } },
  "interaction": { "hover": true, "tooltipDelay": 100, "hideEdgesOnDrag": false }
}
    """)


    # Instead of writing to disk, write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        path = tmp.name
        net.save_graph(path)
        tmp.close()
        with open(path, "r", encoding="utf-8") as f:
            html_content = f.read()
        os.remove(path)
    return html_content
