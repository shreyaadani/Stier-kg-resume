import networkx as nx
from typing import Dict, List, Tuple

def build_graph(ents: Dict[str, List[str]], edges: List[Tuple[str,str,str]]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for group, items in ents.items():
        for it in items:
            G.add_node(it, group=group, title=f"{group}: {it}")
    for s, r, d in edges:
        G.add_edge(s, d, label=r)
    return G
