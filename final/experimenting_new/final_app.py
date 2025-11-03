import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from erd import make_erd
from graph import make_graph_html
from schema import schema_modelling
from graphDB.final.sqlite_to_csv import export_sqlite_to_csv
from erd_text import erd_text_generate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

st.set_page_config(page_title="Neo4j ERD & Graph Viewer", layout="wide")

# === Sidebar ===
st.sidebar.title("Neo4j Connection")
neo4j_url = st.sidebar.text_input("Neo4j URI", placeholder="neo4j+s://...")
neo4j_user = st.sidebar.text_input("Neo4j Username", placeholder="neo4j")
neo4j_pass = st.sidebar.text_input("Neo4j Password", type="password")

st.sidebar.markdown("---")
generate_erd_btn = st.sidebar.button("Generate ERD")
generate_graph_btn = st.sidebar.button("Generate Graph")

st.title("📊 Database Schema Visualizer")

db_name = "sqlite-sakila"
db_path = db_name + ".sqlite"

# === Initialize session state once ===
if "llm1" not in st.session_state:
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY1"]
    st.session_state.llm1 = ChatGoogleGenerativeAI(
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

if "modelling_output" not in st.session_state:
    replacements = export_sqlite_to_csv(db_path, db_name + "_files")
    schema_info, erd_text = erd_text_generate(db_name, replacements)
    st.session_state.modelling_output = schema_modelling(erd_text, st.session_state.llm1)

# === Generate ERD ===
if generate_erd_btn:
    st.subheader("📑 ERD Diagram")
    erd_img = make_erd(db_path)
    st.image(erd_img, caption="ERD Diagram", use_container_width=True)

# === Generate Graph ===
if generate_graph_btn:
    st.subheader("🔗 Graph View")
    graph_html = make_graph_html(st.session_state.modelling_output)
    with open(graph_html, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)
