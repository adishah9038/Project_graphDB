from sqlalchemy import create_engine, MetaData
from graphviz import Digraph
import tempfile
import os

def make_erd(db_path): # uploads/sales_master.sqlite
    engine = create_engine(f"sqlite:///{db_path}")
    metadata = MetaData()
    metadata.reflect(bind=engine) # Reflects its tables, columns, and relationships (like foreign keys) into a MetaData object.

    dot = Digraph(comment="Entity-Relationship Diagram", format="png") # Initializes a directed graph (Digraph)
    dot.attr(rankdir="LR", fontsize="10", nodesep="1", ranksep="2")
    # Sets layout attributes:
    #   1. rankdir="LR" → draw from Left → Right (instead of top-down)
    #   2. nodesep and ranksep control spacing.

    for table in metadata.tables.values():
        columns = "\n".join([f"{col.name} ({col.type})" for col in table.columns])
        dot.node(table.name, f"{table.name}\n{columns}", shape="box")
        for fk in table.foreign_keys:
            dot.edge(table.name, fk.column.table.name, label=fk.column.name)
    # For each table:
    #   1. Builds a list of columns (name (type)).
    #   2. Adds a box-shaped node showing table name + columns.
    #   3. For each foreign key, draws an edge between related tables.

    tmpdir = tempfile.mkdtemp()
    # C:\Users\<YourName>\AppData\Local\Temp\tmpabcd1234 
    outpath = os.path.join(tmpdir, "erd_diagram")
    # C:\Users\<YourName>\AppData\Local\Temp\tmpabcd1234\erd_diagram
    output_file = dot.render(outpath, format="png", cleanup=True)
    # renders image in the temperory location
    return output_file

    # tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    # dot.render(tmpfile.name, cleanup=True)
    # return tmpfile.name + ".png"