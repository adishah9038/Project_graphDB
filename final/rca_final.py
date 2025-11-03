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

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
        [
            (
"system",
"""
You are a expert data analyst. Your job is to do the root cause analysis for the provided question. 
You will be provided with a graph database schema and a graph query_tool to query the data from graph database. 
Note:
1. query_tool accepts instructions in english language only.
"""
            ),
            (
"human",
"""
### Graph database schema (Use it for understanding relations)
---
{schema}
---
### Conversation History:
---
{conversation}
---
User Question:
{question}

"""
            ),
        ]
    )
chain = prompt | llm_with_tool

graph_builder = StateGraph(State)

# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [chain.invoke({"schema":graph_db.schema,"conversation":state['messages'], "question": state['question'],"loop_count":state["loop_count"]+1})]}

graph_builder.add_node("chatbot", chatbot)

from langchain_core.prompts import ChatPromptTemplate
summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You job is to summarize the conversation and frame an answer for the main question asked in the conversation."),
            ("human","{conversation}")
        ]
    )
summary_chain = summary_prompt | llm
def summary(state: State): 
    return {"answer":summary_chain.invoke({"conversation":state['messages']})}

graph_builder.add_node("summary", summary)

import json

from langchain_core.messages import ToolMessage

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[query_tool])
graph_builder.add_node("tools", tool_node)

def route_tools(state: State,):
    if state['loop_count']>10:
        return "summary"
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END, "summary":"summary"},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("summary",END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    """
    Generator version that yields messages one by one as they are generated.
    Each yield is a JSON string ready to be sent to frontend.
    """
    initial_state = {
        "question": user_input,
        "messages": [],
        "loop_count": 0,
        "answer": "",
    }

    yield json.dumps({"type": "system", "content": f"🧠 Processing user input: {user_input}"})

    for event in graph.stream(initial_state):
        for value in event.values():
            messages = value.get("messages", [])
            if not messages:
                continue

            last_msg = messages[-1]
            role = getattr(last_msg, "type", "ai")
            content = getattr(last_msg, "content", "")

            # 🧍 User messages
            if role in ["human", "user"]:
                yield json.dumps({"type": "user", "content": content})

            # 🤖 AI messages
            elif role in ["ai", "assistant"]:
                if content.strip():
                    yield json.dumps({"type": "ai", "content": content})
                else:
                    tool_calls = getattr(last_msg, "tool_calls", None)
                    if tool_calls:
                        for call in tool_calls:
                            tool_args = call.get("args", {})
                            query_text = tool_args.get("query", "").strip()
                            if query_text:
                                yield json.dumps({"type": "tool", "query": query_text})

            # 🧰 Tool message fallback
            elif role == "tool":
                tool_args = getattr(last_msg, "args", {})
                query_text = tool_args.get("query", "").strip()
                if query_text:
                    yield json.dumps({"type": "tool", "query": query_text})

    # finally yield the full final state
    yield json.dumps({"type": "system", "content": "🟢 RCA Complete"})


# # Example usage
# user_input = "Why is the sale decreasing from 2004 to 2005?"
# final_state = stream_graph_updates(user_input)

# print("🧩 Final Agent State Keys:", list(final_state.keys()))
# print("💬 Final Answer:", final_state.get("answer", "No answer found"))
