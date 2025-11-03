from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import os

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY4"]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
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

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    question:str
    messages:Annotated[list,add_messages]
    loop_count:int
    answer:str

    # Wait 60 seconds before connecting using these details, or login to https://console.neo4j.io to validate the Aura Instance is available
NEO4J_URI="neo4j+s://2d5e8539.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="xn8iCGEj2vymA3-43-57qlL63CD70SthzTE_Mt8QfG0"
NEO4J_DATABASE="neo4j"
AURA_INSTANCEID="2d5e8539"
AURA_INSTANCENAME="Instance01"


from langchain_neo4j import Neo4jGraph
enhanced_graph_db = Neo4jGraph(
    url=NEO4J_URI,
    username="neo4j",
    password=NEO4J_PASSWORD,
    driver_config={
        "max_connection_lifetime": 300,  # 5 minutes
        "keep_alive": True,
        "max_connection_pool_size": 50
    },
    enhanced_schema=True)

graph_db = Neo4jGraph(
    url=NEO4J_URI,
    username="neo4j",
    password=NEO4J_PASSWORD,
    driver_config={
        "max_connection_lifetime": 300,  # 5 minutes
        "keep_alive": True,
        "max_connection_pool_size": 50
    },
    enhanced_schema=False)

from langraph_neo4j3 import AgentState, run_agent_workflow
from langchain_core.tools import tool

@tool
def query_tool(query):
    """This tool can query data from graph database. Query must be in english only."""
    state: AgentState = {
            "question": query,
            "next_action": "",
            "cypher_errors": [],
            "database_records": [],
            "steps": [],
            "answer": "",
            "cypher_statement": ""
        }
    result = run_agent_workflow(state,enhanced_graph_db)
    return result["answer"]

tools=[query_tool]
llm_with_tool=llm.bind_tools(tools)
llm_with_tool

