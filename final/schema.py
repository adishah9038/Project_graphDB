import sqlite3
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os

def schema_modelling(erd_text,llm):
    
    graph_schema_prompt = """
    You are an expert at Graph database schema modelling.
    You will be given a entity-relationship markdown.

    Your task is to:
    1. Parse the ERD and identify tables as **graph nodes**. Note that there can be look-up tables that should be identified as **graph relationship**. Look-up tables can contain multiple foreign keys or can also have a file nomenclature with multiple entities.
    2. Extract **graph relationships** from primary keys (PK) and foreign keys (FK).
    3. All identified **graph nodes** should have:
    - name: The entity name (capitalized table name without _table)
    - key: The column that uniquely identify this node
    - properties: List of column names that represent descriptive attributes of this node. 
    - table_name: Name of the table from which the node is extracted

    4. All identified **graph relationships** should have:
    - label: A short label for the relationship (e.g., OWNS, BELONGS_TO, PURCHASED). Donot repeat labels in graph schema
    - source_node: The source node name
    - target_node: The target node name
    - key_s: The primary key from source node only 
    - key_t: The primary key from target node only
    - properties: Columns that describe attributes of the relationship. Properties are likely to be present in case of look-up tables.
    - table_name: Should be the source table name if the relationship has been identified from foreign keys, else it should be the look-up table name.

    5. Return the output as a JSON-like structure with exactly two lists: **nodes** and **relationships**.

    Make sure all tables and relationships in the ERD are represented.
    """

    from langchain_core.prompts import ChatPromptTemplate
    modelling_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", graph_schema_prompt),
            ("human", 
                "ERD Markdown: {{ERD}}"
            ),
        ],
        template_format="jinja2",
    )

    from pydantic import BaseModel, Field
    from typing import List
    from pydantic import BaseModel, Field

    class Node(BaseModel):
        name: str = Field(description="The name of the node, capitalized")
        key: str = Field(description="Primary attribute of the node")
        properties: List[str] = Field(description="Attributes of the node")
        table_name: List[str] = Field(description="List of table names")

    class Relationship(BaseModel):
        source: str = Field(description="Source node name")
        target: str = Field(description="Target node name")
        label: str = Field(description="Type/label of the relationship")
        key_s: str = Field(description="Primary key of source node")
        key_t: str = Field(description="Primary key of target node")
        properties: List[str] = Field(description="Attributes of the relationship")
        table_name: str = Field(description="Table name from which relationship is extracted")

    class ModellingOutput(BaseModel):
        nodes: List[Node] = Field(description="List of node definitions")
        relationships: List[Relationship] = Field(description="List of relationship definitions")

    # ---- chain ----
    modelling_chain = modelling_prompt | llm.with_structured_output(ModellingOutput)

    modelling_output = modelling_chain.invoke(
        {
            "ERD":erd_text,
        }
    )
    return modelling_output