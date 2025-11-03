from pyvis.network import Network
import tempfile

def make_graph_html(modelling_output):
    net = Network(
        notebook=False,
        directed=True,
        height="700px", 
        width="100%", 
        bgcolor="#ffffff",
        font_color="black"
    )

    # === Add nodes ===
    for node in modelling_output.nodes:
        props_preview = ", ".join(node.properties[:5]) + ("..." if len(node.properties) > 5 else "")
        label = (
            f"{node.name}\n"
            f"PK: {node.key}\n"
            f"Props: {props_preview}"
        )
        net.add_node(
            node.name,
            label=label,
            color="lightblue",
            shape="box",
            title=f"Table: {', '.join(node.table_name)}<br>Properties: {', '.join(node.properties)}"
        )

    # === Add relationships ===
    for rel in modelling_output.relationships:
        props_text = f" ({', '.join(rel.properties)})" if rel.properties else ""
        net.add_edge(
            rel.source,
            rel.target,
            label=rel.label,
            title=f"{rel.table_name}<br>{rel.key_s} → {rel.key_t}{props_text}",
            width=2
        )

    # === Graph layout options ===
    net.set_options("""
    {
    "physics": {
        "barnesHut": {
        "gravitationalConstant": -30000,
        "centralGravity": 0.005,
        "springLength": 300,
        "springConstant": 0.02
        },
        "stabilization": { "iterations": 250 }
    },
    "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "tooltipDelay": 200
    },
    "edges": {
        "smooth": true
    }
    }
    """)

    # === Save to temp HTML ===
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmpfile.name)
    return tmpfile.name
