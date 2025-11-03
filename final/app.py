from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from inject_relationship import execute_ultra_optimized_relationships
from inject_node import execute_ultra_optimized_nodes
from schema import schema_modelling
from sqlite_to_csv import export_sqlite_to_csv
from erd_text import erd_text_generate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from neo4j import GraphDatabase
from langraph_neo4j3 import AgentState, run_agent_workflow
from langchain_neo4j import Neo4jGraph

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) 

# Initialize LLM
os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY4"]
llm = ChatGoogleGenerativeAI(
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

driver = None
db_name = None
modelling_output = None
neo4j_graph = None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload_sqlite', methods=['POST'])
def upload_sqlite():
    global db_name
    file = request.files['sqlite_file']
    if file:
        filename = secure_filename(file.filename)
        db_name = os.path.splitext(filename)[0]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({"status": "success", "db_name": db_name})
    return jsonify({"status": "error"})

from flask import send_file
from erd import make_erd  

@app.route('/generate_erd', methods=['POST'])
def generate_erd():
    global db_name
    erd_img_path = make_erd(f"uploads/{db_name}.sqlite")  
    return jsonify({"erd_img_url": f"/show_erd?img={erd_img_path}"})

@app.route('/show_erd')
def show_erd():
    img_path = request.args.get("img")
    return send_file(img_path, mimetype='image/png')

@app.route('/connect_graph', methods=['POST'])
def connect_graph():
    global driver, neo4j_graph
    data = request.json
    uri = data['uri']
    username = data['username']
    password = data['password']
    driver = GraphDatabase.driver(uri, auth=(username, password))
    neo4j_graph = Neo4jGraph(
        url=uri,
        username=username,
        password=password,
        database="neo4j", 
        driver_config={
            "max_connection_lifetime": 3000, 
            "keep_alive": True,
            "max_connection_pool_size": 50
        },
        enhanced_schema=True)
    return jsonify({"status": "connected"})

from graph import make_graph_html

@app.route('/generate_schema', methods=['POST'])
def generate_schema():
    global modelling_output, db_name
    if not db_name:
        return jsonify({"error": "No database uploaded"}), 400
    replacement = export_sqlite_to_csv(db_name+".sqlite",db_name+"_files")
    _, erd_text = erd_text_generate(db_name, replacement)
    modelling_output = schema_modelling(erd_text, llm)
    graph_html_path = make_graph_html(modelling_output)
    return jsonify({
        "graph_html_url": f"/show_graph?html={graph_html_path}"
    })

@app.route('/show_graph')
def show_graph():
    html_path = request.args.get("html")
    return send_file(html_path, mimetype='text/html')

@app.route('/inject_nodes', methods=['POST'])
def inject_nodes():
    global modelling_output, driver, db_name
    stats = execute_ultra_optimized_nodes(modelling_output.nodes, driver, db_name+"_files")
    return jsonify({"status": "nodes_injected", "stats": str(stats)})

@app.route('/inject_relationships', methods=['POST'])
def inject_relationships():
    global modelling_output, driver, db_name
    results = execute_ultra_optimized_relationships(
        relationships=modelling_output.relationships,
        driver=driver,
        output_dir=db_name+"_files",
        nodes_list=modelling_output.nodes,
        initial_batch_size=1000,
        max_workers=6
    )
    return jsonify({"status": "relationships_injected", "results": str(results)})

@app.route('/query_graph', methods=['POST'])
def query_graph():
    global driver,neo4j_graph
    data = request.json
    user_input = data['question']
    state: AgentState = {
        "question": user_input,
        "next_action": "",
        "cypher_errors": [],
        "database_records": [],
        "steps": [],
        "answer": "",
        "cypher_statement": ""
    }
    result = run_agent_workflow(state, neo4j_graph)
    return jsonify({"answer": result["answer"], "cypher": result["cypher_statement"]})

@app.route('/test_query')
def test_query():
    with driver.session(database="neo4j") as session:
        result = session.run("RETURN 1 AS n")
        return jsonify({"result": [r["n"] for r in result]})

from flask import Response, stream_with_context, request
from rca_final import stream_graph_updates

@app.route('/stream_rca', methods=['POST'])
def stream_rca():
    user_input = request.json.get("question", "")

    def generate():
        for msg in stream_graph_updates(user_input):
            yield msg + "\n"  # newline is important for frontend to split
    return Response(stream_with_context(generate()), mimetype="text/plain")

if __name__ == '__main__':
    app.run(debug=True)
