from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import os

# Define two LLMs with different API keys
os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY1"]
validate_cypher_llm = ChatGoogleGenerativeAI(
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

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY2"]
text2cypher_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
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

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY3"]
llm3 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=1,
)
guard_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1,
)

from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated, TypedDict
from operator import add

class AgentState(TypedDict):
    question: str
    next_action: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]
    answer: str
    cypher_statement: str

def run_agent_workflow(state: AgentState,neo4j_graph):

    # from langchain_neo4j import Neo4jGraph
    # graph = Neo4jGraph(
    #     url=NEO4J_URL,
    #     username="neo4j",
    #     password=NEO4J_PASSWORD,
    #     driver_config={
    #         "max_connection_lifetime": 300,  # 5 minutes
    #         "keep_alive": True,
    #         "max_connection_pool_size": 50
    #     })

    # enhanced_graph = Neo4jGraph(
    #     url=NEO4J_URL,
    #     username="neo4j",
    #     password=NEO4J_PASSWORD,
    #     driver_config={
    #         "max_connection_lifetime": 300,  # 5 minutes
    #         "keep_alive": True,
    #         "max_connection_pool_size": 50
    #     },
    #     enhanced_schema=True)
    enhanced_graph = neo4j_graph

    from typing import Literal
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    from langchain_core.output_parsers import StrOutputParser
    import re
    from langchain_core.messages import AIMessage

    schema = enhanced_graph.schema
    # guardrails_system = f"""
    # As an intelligent assistant, your primary objective is to decide whether a given question is related to the database described below.

    # Database Schema:
    # {schema}

    # If the question is is related to the schema, output "data" else output "end".
    # Your output must be one of the following: "data" or "end".
    # """

    # guardrails_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             guardrails_system,
    #         ),
    #         (
    #             "human",
    #             ("{question}"),
    #         ),
    #     ]
    # )

    # class GuardrailsOutput(BaseModel):
    #     decision: Literal["data", "end"] = Field(
    #         description="Decision on whether the question is related to data"
    #     )

    # guardrails_chain = guardrails_prompt | guard_llm.with_structured_output(GuardrailsOutput)

    # def guardrails(state: AgentState) -> AgentState:
    #     guardrails_output = guardrails_chain.invoke({"question": state.get("question")})

    #     return {
    #         "question": state["question"],
    #         "next_action": guardrails_output.decision,
    #         "cypher_statement": "",
    #         "cypher_errors": [],
    #         "database_records": [],
    #         "steps": ["guardrails"],
    #         "answer": (
    #             "This question is not about the data. Therefore, I cannot answer it."
    #             if guardrails_output.decision == "end"
    #             else ""
    #         ),
    #     }

    from langchain_core.output_parsers import StrOutputParser
    import re

    text2cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert Neo4j Cypher query generator. 
    Your task is to take a natural language user question and produce a single valid Cypher query. 

    ### Schema
    ---
    {schema}
    ---
    """
            ),
            (
                "human",
        """User Question:
    {question}

    Respond in the format:

    <cypher>
    MATCH ...
    </cypher>

    """
            ),
        ]
    )
    from langchain_core.messages import AIMessage

    def extract_cypher(ai_message):
        """
        Takes the AIMessage output from text2cypher_chain and extracts a valid Cypher query string.
        """
        # If AIMessage, get its content
        text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        
        # Use regex to find text inside <cypher>...</cypher>
        match = re.search(r"<cypher>\s*(.*?)\s*</cypher>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # If no <cypher> tags found, return the text as is
        return text.strip()


    text2cypher_chain = text2cypher_prompt | text2cypher_llm


    def generate_cypher(state: AgentState) -> AgentState:
        """
        Generates a Cypher statement based on few-shot examples and schema context.
        """

        generated_cypher = text2cypher_chain.invoke(
            {
                "question": state.get("question"),
                "schema": enhanced_graph.schema,
            }
        )
        plain_cypher = extract_cypher(generated_cypher)
        
        return {
            **state,
            "cypher_statement": plain_cypher,
            "steps": ["generate_cypher"],
            "next_action":"validate_cypher"
        }

    from typing import List, Optional

    validate_cypher_system = """
    You are a Cypher expert reviewing a statement written by a junior developer.
    """

    validate_cypher_user = """
    You are validating a Cypher statement against a graph schema.

    Your tasks:
    1. Identify all syntax or semantic errors in the Cypher statement.
    2. Extract all property filters used in the query.  
    A property filter is any condition on a node or relationship property, including:
    - Direct matches in `property: value`
    - Conditions in WHERE clauses (`=`, `<>`, `>`, `<`, `>=`, `<=`, `IN`, `CONTAINS`, etc.)

    Output format:
    - `errors`: list of strings describing any issues (empty if none).
    - `filters`: list of objects with `node_label`, `property_key`, `property_value`.  

    Schema:
    {schema}

    Question:
    {question}

    Cypher:
    {cypher}

    Be very careful: do not miss any filter inside MATCH or WHERE clauses.
    """

    validate_cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                validate_cypher_system,
            ),
            (
                "human",
                (validate_cypher_user),
            ),
        ]
    )

    class Property(BaseModel):
        """
        Represents a filter condition based on a specific node property in a graph in a Cypher statement.
        """

        node_label: str = Field(
            description="The label of the node to which this property belongs."
        )
        property_key: str = Field(description="The key of the property being filtered.")
        property_value: str = Field(
            description="The value that the property is being matched against."
        )

    class ValidateCypherOutput(BaseModel):
        """
        Represents the validation result of a Cypher query's output,
        including any errors and applied filters.
        """

        errors: Optional[List[str]] = Field(
            description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
        )
        filters: Optional[List[Property]] = Field(
            description="A list of property-based filters applied in the Cypher statement."
        )


    validate_cypher_chain = validate_cypher_prompt | validate_cypher_llm.with_structured_output(
        ValidateCypherOutput
    )

    from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

    # Cypher query corrector is experimental
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in enhanced_graph.structured_schema.get("relationships")
    ]
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    from neo4j.exceptions import CypherSyntaxError

    def validate_cypher(state: AgentState) -> AgentState:
        """
        Validates the Cypher statements and maps any property values to the database.
        """
        errors = []
        mapping_errors = []

        cypher = state.get("cypher_statement", "")
        question = state.get("question", "")

        # 1. Syntax validation via EXPLAIN
        try:
            enhanced_graph.query(f"EXPLAIN {cypher}")
        except CypherSyntaxError as e:
            errors.append(e.message)

        # 2. Relationship direction correction
        corrected_cypher = cypher_query_corrector(cypher)
        if not corrected_cypher:
            errors.append("The generated Cypher statement doesn't fit the graph schema")
        elif corrected_cypher != cypher:
            print("Relationship direction was corrected")

        # 3. LLM-based validation
        llm_output = validate_cypher_chain.invoke(
            {
                "question": question,
                "schema": enhanced_graph.schema,
                "cypher": corrected_cypher,
            }
        )
        if llm_output.errors:
            errors.extend(llm_output.errors)

        # 4. Filter value mapping
        if llm_output.filters:
            for filter in llm_output.filters:
                # Do mapping only for string values
                if (
                    not [
                        prop
                        for prop in enhanced_graph.structured_schema["node_props"][
                            filter.node_label
                        ]
                        if prop["property"] == filter.property_key
                    ][0]["type"]
                    == "STRING"
                ):
                    continue
                mapping = enhanced_graph.query(
                    f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                    {"value": filter.property_value},
                )
                if not mapping:
                    print(
                        f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                    )
                    mapping_errors.append(
                        f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                    )

        # 5. Decide next action
        if mapping_errors:
            next_action = "end"
            answer = "This question refers to a value that does not exist in the data."
        elif errors:
            next_action = "correct_cypher"
            answer = ""
        else:
            next_action = "execute_cypher"
            answer = ""

        # 6. Return new AgentState
        return {
            "question": question,
            "next_action": next_action,
            "cypher_statement": corrected_cypher,
            "cypher_errors": errors,
            "database_records": state.get("database_records", []),
            "steps": ["validate_cypher"],
            "answer": answer,
        }

    correct_cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Cypher expert reviewing a statement written by a junior developer. "
                    "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                    "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
                ),
            ),
            (
                "human",
                (
                    """Check for invalid syntax or semantics and return a corrected Cypher statement.

    Schema:
    {schema}

    Note: Do not include any explanations or apologies in your responses.
    Do not wrap the response in any backticks or anything else.
    Respond with a Cypher statement only!

    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    The errors are:
    {errors}

    Corrected Cypher statement: """
                ),
            ),
        ]
    )

    correct_cypher_chain = correct_cypher_prompt | llm3 | StrOutputParser()

    def correct_cypher(state: AgentState) -> AgentState:
        """
        Correct the Cypher statement based on the provided errors.
        """
        corrected_cypher = correct_cypher_chain.invoke(
            {
                "question": state.get("question"),
                "errors": state.get("cypher_errors"),
                "cypher": state.get("cypher_statement"),
                "schema": enhanced_graph.schema,
            }
        )

        return {
            "next_action": "validate_cypher",
            "cypher_statement": corrected_cypher,
            "steps": ["correct_cypher"],
        }

    no_results = "I couldn't find any relevant information in the database"

    def execute_cypher(state: AgentState) -> AgentState:
        """
        Executes the given Cypher statement.
        """

        records = enhanced_graph.query(state.get("cypher_statement"))
        return {
            "database_records": records if records else no_results,
            "next_action": "generate_final_answer",
            "steps": ["execute_cypher"],
        }

    generate_final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant",
            ),
            (
                "human",
                (
                    """Use the following results retrieved from a database to provide
    a succinct, definitive answer to the user's question.

    Respond as if you are answering the question directly.

    Results: {results}
    Question: {question}"""
                ),
            ),
        ]
    )

    generate_final_chain = generate_final_prompt | llm3 | StrOutputParser()

    def generate_final_answer(state: AgentState) -> AgentState:
        if(state['steps'][-1]=='guardrails'): 
            print("Question out of context.")
            return {
                **state,
                "steps":["generate_final_answer"],
            }
        """
        Generates a final answer.
        """
        final_answer = generate_final_chain.invoke(
            {"question": state.get("question"), "results": state.get("database_records")}
        )
        return {"answer": final_answer, "steps": ["generate_final_answer"],"next_action":"END"}

    from IPython.display import Image, display
    from langgraph.graph import END, START, StateGraph

    langgraph = StateGraph(AgentState)
    # langgraph.add_node(guardrails)
    langgraph.add_node(generate_cypher)
    langgraph.add_node(validate_cypher)
    langgraph.add_node(correct_cypher)
    langgraph.add_node(execute_cypher)
    langgraph.add_node(generate_final_answer)

    # langgraph.add_edge(START, "guardrails")
    # def guardrails_condition(
    #     state: AgentState,
    # ) -> Literal["generate_cypher", "generate_final_answer"]:
    #     if state.get("next_action") == "end":
    #         return "generate_final_answer"
    #     elif state.get("next_action") == "data":
    #         return "generate_cypher"
    #     else:
    #         raise ValueError(f"Unexpected next_action: {state.get('next_action')}")
    # langgraph.add_conditional_edges(
    #     "guardrails",
    #     guardrails_condition,
    # )
    langgraph.add_edge(START,"generate_cypher")
    langgraph.add_edge("generate_cypher", "validate_cypher")
    def validate_cypher_condition(
        state: AgentState,
    ) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
        if state.get("next_action") == "end":
            return "generate_final_answer"
        elif state.get("next_action") == "correct_cypher":
            return "correct_cypher"
        elif state.get("next_action") == "execute_cypher":
            return "execute_cypher"
    langgraph.add_conditional_edges(
        "validate_cypher",
        validate_cypher_condition,
    )
    langgraph.add_edge("execute_cypher", "generate_final_answer")
    langgraph.add_edge("correct_cypher", "validate_cypher")
    langgraph.add_edge("generate_final_answer", END)

    agent = langgraph.compile()

    final_state = agent.invoke(state)
    return final_state

# initial_agent_state: AgentState = {
#     "messages":[],
#     "question": "What is the sale for France for 2003?",
#     "next_action": "",
#     "cypher_statement": "",
#     "cypher_errors": [],
#     "database_records": [],
#     "steps": [],
#     "answer": ""
# }


    