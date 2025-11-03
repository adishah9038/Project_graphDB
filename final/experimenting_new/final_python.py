from inject_relationship import execute_ultra_optimized_relationships
from inject_node import execute_ultra_optimized_nodes
from schema import schema_modelling
from graphDB.final.sqlite_to_csv import export_sqlite_to_csv
from erd_text import erd_text_generate

from erd import make_erd
from graph import make_graph_html

# ENV
from dotenv import load_dotenv
load_dotenv()
#LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import os

# Define two LLMs with different API keys
os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY1"]
llm1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=1,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

from neo4j import GraphDatabase

NEO4J_URI="neo4j+s://eaee53dc.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="DH_4xGGIxkB2Acm6lfRoU6FOdYbW2bcGga2b4yLoIQE"
NEO4J_DATABASE="neo4j"
AURA_INSTANCEID="eaee53dc"
AURA_INSTANCENAME="Free instance"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
print(driver)

db_name = "sqlite-sakila"
db_path = db_name+".sqlite"

erd_path = make_erd(db_path)

replacements = export_sqlite_to_csv(db_name+".sqlite",db_path)

schema_info, erd_text = erd_text_generate(db_name, replacements)

modelling_output = schema_modelling(erd_text,llm1)

graph_path = make_graph_html(modelling_output)
with open(graph_path, "r", encoding="utf-8") as f:
    html_content = f.read()

inject_node_stats = execute_ultra_optimized_nodes(modelling_output.nodes,driver,db_name+"_files")

results = execute_ultra_optimized_relationships(
    relationships=modelling_output.relationships,
    driver=driver,
    output_dir=db_name+"_files",
    nodes_list=modelling_output.nodes,  # Pass your nodes list
    initial_batch_size=1000,  # Smaller for debugging
    max_workers=6  # Reduced for stability
)

from langraph_neo4j3 import AgentState, run_agent_workflow
user_input ="What is the total revenue per year?"

state: AgentState = {
        "question": user_input,
        "next_action": "",
        "cypher_errors": [],
        "database_records": [],
        "steps": [],
        "answer": "",
        "cypher_statement": ""
    }

result = run_agent_workflow(state,"neo4j+s://eaee53dc.databases.neo4j.io","DH_4xGGIxkB2Acm6lfRoU6FOdYbW2bcGga2b4yLoIQE")