from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from neo4j import GraphDatabase
import logging
import traceback

# Import your existing modules
from inject_relationship import execute_ultra_optimized_relationships
from inject_node import execute_ultra_optimized_nodes
from schema import schema_modelling
from graphDB.final.sqlite_to_csv import export_sqlite_to_csv
from erd_text import erd_text_generate
from langraph_neo4j3 import AgentState, run_agent_workflow

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store state
current_database = None
current_driver = None
current_modelling_output = None
current_schema_info = None

# Initialize LLM
def get_llm():
    try:
        return ChatGoogleGenerativeAI(
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
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_sqlite', methods=['POST'])
def upload_sqlite():
    global current_database
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.sqlite'):
            return jsonify({'error': 'File must be a SQLite database'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store database name without extension
        current_database = os.path.splitext(filename)[0]
        
        logger.info(f"SQLite file uploaded successfully: {filename}")
        return jsonify({
            'success': True, 
            'message': 'SQLite file uploaded successfully',
            'filename': filename
        })
    
    except Exception as e:
        logger.error(f"Error uploading SQLite file: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/generate_erd', methods=['POST'])
def generate_erd():
    global current_database, current_schema_info
    try:
        if not current_database:
            return jsonify({'error': 'No SQLite file uploaded'}), 400
        
        db_path = os.path.join(app.config['UPLOAD_FOLDER'], current_database + '.sqlite')
        output_dir = os.path.join('static/outputs', current_database + '_files')
        
        # Export SQLite to CSV
        replacements = export_sqlite_to_csv(db_path, output_dir)
        
        # Generate ERD text
        schema_info, erd_text = erd_text_generate(current_database, replacements)
        current_schema_info = schema_info
        
        # Save ERD text to file
        erd_file = os.path.join('static/outputs', current_database + '_erd.txt')
        with open(erd_file, 'w') as f:
            f.write(erd_text)
        
        logger.info("ERD generated successfully")
        return jsonify({
            'success': True,
            'message': 'ERD generated successfully',
            'erd_text': erd_text,
            'schema_info': schema_info
        })
    
    except Exception as e:
        logger.error(f"Error generating ERD: {str(e)}")
        return jsonify({'error': f'ERD generation failed: {str(e)}'}), 500

@app.route('/generate_schema', methods=['POST'])
def generate_schema():
    global current_database, current_modelling_output, current_schema_info
    try:
        if not current_database or not current_schema_info:
            return jsonify({'error': 'ERD must be generated first'}), 400
        
        # Read ERD text
        erd_file = os.path.join('static/outputs', current_database + '_erd.txt')
        with open(erd_file, 'r') as f:
            erd_text = f.read()
        
        # Get LLM
        llm = get_llm()
        if not llm:
            return jsonify({'error': 'LLM not available'}), 500
        
        # Generate schema modeling
        modelling_output = schema_modelling(erd_text, llm)
        current_modelling_output = modelling_output
        
        # Convert to JSON serializable format
        schema_data = {
            'nodes': [{'name': node.name, 'properties': node.properties} for node in modelling_output.nodes],
            'relationships': [
                {
                    'source': rel.source,
                    'target': rel.target,
                    'relationship': rel.relationship,
                    'properties': rel.properties
                } for rel in modelling_output.relationships
            ]
        }
        
        # Save schema to file
        schema_file = os.path.join('static/outputs', current_database + '_schema.json')
        with open(schema_file, 'w') as f:
            json.dump(schema_data, f, indent=2)
        
        logger.info("Schema generated successfully")
        return jsonify({
            'success': True,
            'message': 'Graph schema generated successfully',
            'schema': schema_data
        })
    
    except Exception as e:
        logger.error(f"Error generating schema: {str(e)}")
        return jsonify({'error': f'Schema generation failed: {str(e)}'}), 500

@app.route('/connect_neo4j', methods=['POST'])
def connect_neo4j():
    global current_driver
    try:
        data = request.get_json()
        uri = data.get('uri')
        username = data.get('username')
        password = data.get('password')
        
        if not all([uri, username, password]):
            return jsonify({'error': 'All connection parameters are required'}), 400
        
        # Test connection
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        current_driver = driver
        
        logger.info("Neo4j connection established successfully")
        return jsonify({
            'success': True,
            'message': 'Neo4j connection established successfully'
        })
    
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {str(e)}")
        return jsonify({'error': f'Neo4j connection failed: {str(e)}'}), 500

@app.route('/inject_nodes', methods=['POST'])
def inject_nodes():
    global current_driver, current_modelling_output, current_database
    try:
        if not current_driver:
            return jsonify({'error': 'Neo4j connection not established'}), 400
        
        if not current_modelling_output:
            return jsonify({'error': 'Schema must be generated first'}), 400
        
        output_dir = os.path.join('static/outputs', current_database + '_files')
        
        # Execute node injection
        inject_stats = execute_ultra_optimized_nodes(
            current_modelling_output.nodes, 
            current_driver, 
            output_dir
        )
        
        logger.info("Nodes injected successfully")
        return jsonify({
            'success': True,
            'message': 'Nodes injected successfully',
            'stats': str(inject_stats)
        })
    
    except Exception as e:
        logger.error(f"Error injecting nodes: {str(e)}")
        return jsonify({'error': f'Node injection failed: {str(e)}'}), 500

@app.route('/inject_relationships', methods=['POST'])
def inject_relationships():
    global current_driver, current_modelling_output, current_database
    try:
        if not current_driver:
            return jsonify({'error': 'Neo4j connection not established'}), 400
        
        if not current_modelling_output:
            return jsonify({'error': 'Schema must be generated first'}), 400
        
        output_dir = os.path.join('static/outputs', current_database + '_files')
        
        # Execute relationship injection
        results = execute_ultra_optimized_relationships(
            relationships=current_modelling_output.relationships,
            driver=current_driver,
            output_dir=output_dir,
            nodes_list=current_modelling_output.nodes,
            initial_batch_size=1000,
            max_workers=6
        )
        
        logger.info("Relationships injected successfully")
        return jsonify({
            'success': True,
            'message': 'Relationships injected successfully',
            'results': str(results)
        })
    
    except Exception as e:
        logger.error(f"Error injecting relationships: {str(e)}")
        return jsonify({'error': f'Relationship injection failed: {str(e)}'}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    global current_driver
    try:
        if not current_driver:
            return jsonify({'error': 'Neo4j connection not established'}), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get connection details from current driver
        uri = current_driver._pool.address.host_port()
        # Note: You'll need to store password separately for the agent workflow
        # For security, consider using session-based storage
        
        state = {
            "question": question,
            "next_action": "",
            "cypher_errors": [],
            "database_records": [],
            "steps": [],
            "answer": "",
            "cypher_statement": ""
        }
        
        # You'll need to modify this to pass the correct parameters
        # This assumes you have the password stored securely
        result = run_agent_workflow(
            state, 
            f"neo4j+s://{uri}", 
            "your-stored-password-here"  # Replace with actual password management
        )
        
        logger.info(f"Question answered: {question}")
        return jsonify({
            'success': True,
            'question': question,
            'answer': result.get('answer', 'No answer provided'),
            'cypher_statement': result.get('cypher_statement', ''),
            'steps': result.get('steps', [])
        })
    
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return jsonify({'error': f'Question answering failed: {str(e)}'}), 500

@app.route('/static/outputs/<path:filename>')
def serve_output_files(filename):
    return send_from_directory('static/outputs', filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)