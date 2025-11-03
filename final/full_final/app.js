class GraphBuilderApp {
    constructor() {
        this.currentState = {
            sqliteUploaded: false,
            erdGenerated: false,
            schemaGenerated: false,
            neo4jConnected: false,
            nodesInjected: false,
            relationsInjected: false
        };
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateUI();
    }

    bindEvents() {
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('sqliteFile');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Buttons
        document.getElementById('generateERD').addEventListener('click', this.generateERD.bind(this));
        document.getElementById('generateSchema').addEventListener('click', this.generateSchema.bind(this));
        document.getElementById('connectNeo4j').addEventListener('click', this.connectNeo4j.bind(this));
        document.getElementById('injectNodes').addEventListener('click', this.injectNodes.bind(this));
        document.getElementById('injectRelations').addEventListener('click', this.injectRelations.bind(this));
        document.getElementById('askQuestion').addEventListener('click', this.askQuestion.bind(this));

        // Tabs
        document.querySelectorAll('.tab-button').forEach(tab => {
            tab.addEventListener('click', this.switchTab.bind(this));
        });

        // Enter key for question input
        document.getElementById('questionInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.askQuestion();
            }
        });
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileUpload(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFileUpload(file);
        }
    }

    async handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith('.sqlite') && !file.name.toLowerCase().endsWith('.db')) {
            this.showNotification('Please select a SQLite database file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showLoading('Uploading SQLite file...');
            const response = await fetch('/upload_sqlite', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentState.sqliteUploaded = true;
                this.updateFileInfo(file.name);
                this.updateStatusItem(0, 'success');
                this.showNotification(result.message, 'success');
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.updateUI();
        }
    }

    async generateERD() {
        try {
            this.showLoading('Generating ERD from SQLite database...');
            const response = await fetch('/generate_erd', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentState.erdGenerated = true;
                this.updateStatusItem(1, 'success');
                this.displayERD(result.erd_text);
                this.showNotification(result.message, 'success');
                this.switchTab({ target: { dataset: { tab: 'erd' } } });
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showNotification(`ERD generation failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.updateUI();
        }
    }

    async generateSchema() {
        try {
            this.showLoading('Generating graph schema using AI...');
            const response = await fetch('/generate_schema', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentState.schemaGenerated = true;
                this.updateStatusItem(2, 'success');
                this.displaySchema(result.schema);
                this.showNotification(result.message, 'success');
                this.switchTab({ target: { dataset: { tab: 'schema' } } });
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showNotification(`Schema generation failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.updateUI();
        }
    }

    async connectNeo4j() {
        const uri = document.getElementById('neo4jUri').value.trim();
        const username = document.getElementById('neo4jUsername').value.trim();
        const password = document.getElementById('neo4jPassword').value.trim();

        if (!uri || !username || !password) {
            this.showNotification('Please fill in all Neo4j connection details', 'warning');
            return;
        }

        try {
            this.showLoading('Connecting to Neo4j database...');
            const response = await fetch('/connect_neo4j', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uri, username, password })
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentState.neo4jConnected = true;
                this.updateStatusItem(3, 'success');
                this.showNotification(result.message, 'success');
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showNotification(`Neo4j connection failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.updateUI();
        }
    }

    async injectNodes() {
        try {
            this.showLoading('Injecting nodes into Neo4j...');
            const response = await fetch('/inject_nodes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentState.nodesInjected = true;
                this.updateStatusItem(4, 'success');
                this.showNotification(result.message, 'success');
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showNotification(`Node injection failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.updateUI();
        }
    }

    async injectRelations() {
        try {
            this.showLoading('Injecting relationships into Neo4j...');
            const response = await fetch('/inject_relationships', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const result = await response.json();
            
            if (result.success) {
                this.currentState.relationsInjected = true;
                this.updateStatusItem(4, 'success');
                this.showNotification(result.message, 'success');
                this.enableQASystem();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showNotification(`Relationship injection failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.updateUI();
        }
    }

    async askQuestion() {
        const questionInput = document.getElementById('questionInput');
        const question = questionInput.value.trim();
        
        if (!question) {
            this.showNotification('Please enter a question', 'warning');
            return;
        }

        try {
            this.addQuestionToChat(question);
            questionInput.value = '';
            this.showLoading('Processing your question...');
            
            const response = await fetch('/ask_question', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            const result = await response.json();
            
            if (result.success) {
                this.addAnswerToChat(result.answer, result.cypher_statement, result.steps);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.addAnswerToChat(`Sorry, I encountered an error: ${error.message}`, null, []);
        } finally {
            this.hideLoading();
        }
    }

    switchTab(e) {
        const tabName = e.target.dataset.tab;
        
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(tab => {
            tab.classList.remove('active');
        });
        e.target.classList.add('active');

        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
    }

    updateFileInfo(filename) {
        const fileInfo = document.getElementById('fileInfo');
        fileInfo.innerHTML = `<i class="fas fa-check"></i> ${filename} uploaded successfully`;
        fileInfo.style.display = 'block';
    }

    displayERD(erdText) {
        const erdContent = document.getElementById('erdContent');
        erdContent.innerHTML = `<pre>${erdText}</pre>`;
    }

    displaySchema(schema) {
        const schemaContent = document.getElementById('schemaContent');
        let html = '';

        // Display nodes
        if (schema.nodes && schema.nodes.length > 0) {
            html += `
                <div class="schema-section">
                    <h3><i class="fas fa-circle"></i> Nodes (${schema.nodes.length})</h3>
                    <div class="nodes-grid">
            `;
            
            schema.nodes.forEach(node => {
                html += `
                    <div class="node-card">
                        <div class="node-name">
                            <i class="fas fa-circle"></i>
                            ${node.name}
                        </div>
                        <ul class="properties-list">
                            ${node.properties.map(prop => `<li>${prop}</li>`).join('')}
                        </ul>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }

        // Display relationships
        if (schema.relationships && schema.relationships.length > 0) {
            html += `
                <div class="schema-section">
                    <h3><i class="fas fa-arrows-alt"></i> Relationships (${schema.relationships.length})</h3>
                    <div class="relationships-list">
            `;
            
            schema.relationships.forEach(rel => {
                html += `
                    <div class="relationship-card">
                        <div class="relationship-info">
                            <span>${rel.source}</span>
                            <span class="relationship-arrow">→ ${rel.relationship} →</span>
                            <span>${rel.target}</span>
                        </div>
                        ${rel.properties && rel.properties.length > 0 ? 
                            `<ul class="properties-list">
                                ${rel.properties.map(prop => `<li>${prop}</li>`).join('')}
                            </ul>` : ''
                        }
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }

        schemaContent.innerHTML = html || '<div class="placeholder"><i class="fas fa-project-diagram"></i><p>No schema data available</p></div>';
    }

    addQuestionToChat(question) {
        const chatHistory = document.getElementById('chatHistory');
        
        // Remove placeholder if it exists
        const placeholder = chatHistory.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message';
        messageDiv.innerHTML = `
            <div class="message-question">${question}</div>
        `;
        
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    addAnswerToChat(answer, cypherStatement, steps) {
        const chatHistory = document.getElementById('chatHistory');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message';
        
        let html = `<div class="message-answer">${answer}`;
        
        if (cypherStatement) {
            html += `<div class="cypher-code">${cypherStatement}</div>`;
        }
        
        if (steps && steps.length > 0) {
            html += `<div class="message-meta">
                <i class="fas fa-info-circle"></i>
                Steps: ${steps.join(' → ')}
            </div>`;
        }
        
        html += `</div>`;
        messageDiv.innerHTML = html;
        
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    enableQASystem() {
        document.getElementById('questionInput').disabled = false;
        document.getElementById('askQuestion').disabled = false;
        
        const chatHistory = document.getElementById('chatHistory');
        chatHistory.innerHTML = `
            <div class="chat-message">
                <div class="message-answer">
                    <strong>🎉 System Ready!</strong><br>
                    Your graph database is now ready for questions. Try asking something like:
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>"What is the total revenue per year?"</li>
                        <li>"Show me all customers from a specific city"</li>
                        <li>"What are the top-selling products?"</li>
                    </ul>
                </div>
            </div>
        `;
    }

    updateStatusItem(index, status) {
        const statusItems = document.querySelectorAll('.status-item i');
        if (statusItems[index]) {
            statusItems[index].className = `fas fa-circle text-${status}`;
        }
    }

    updateUI() {
        // Update button states
        document.getElementById('generateERD').disabled = !this.currentState.sqliteUploaded;
        document.getElementById('generateSchema').disabled = !this.currentState.erdGenerated;
        document.getElementById('injectNodes').disabled = !this.currentState.schemaGenerated || !this.currentState.neo4jConnected;
        document.getElementById('injectRelations').disabled = !this.currentState.nodesInjected || !this.currentState.neo4jConnected;
    }

    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const text = document.getElementById('loadingText');
        text.textContent = message;
        overlay.classList.add('show');
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.remove('show');
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        notification.innerHTML = `
            <i class="fas ${icons[type] || icons.info}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
        
        // Remove on click
        notification.addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new GraphBuilderApp();
});