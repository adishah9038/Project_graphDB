// ===== Utility Functions =====
function setOutput(content) {
    document.getElementById("output").innerHTML = content;
}

function addMessage(content, sender = "system") {
    let msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.innerHTML = content;
    document.getElementById("qa-messages").appendChild(msgDiv);

    // Auto-scroll to bottom
    let qaMessages = document.getElementById("qa-messages");
    qaMessages.scrollTop = qaMessages.scrollHeight;
}

// ===== Upload DB =====
document.getElementById("uploadForm").onsubmit = async function (e) {
    e.preventDefault();
    let formData = new FormData(this);
    let res = await fetch("/upload_sqlite", { method: "POST", body: formData });
    let data = await res.json();

    if (data.status === "success") {
        addMessage("✅ Database uploaded and set: " + data.db_name, "system");
    } else {
        addMessage("❌ Failed to upload database", "system");
    }
};

// ===== Connect Graph (Neo4j) =====
document.getElementById("connectBtn").addEventListener("click", async () => {
    let uri = document.getElementById("uri").value;
    let username = document.getElementById("username").value;
    let password = document.getElementById("password").value;

    try {
        let res = await fetch("/connect_graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ uri, username, password }),
        });

        let data = await res.json();

        if (data.status === "connected") {
            addMessage("✅ Neo4j graph and driver created successfully.", "system");
        } else {
            addMessage("❌ Failed to create Neo4j graph/driver.", "system");
        }
    } catch (error) {
        addMessage("❌ Error: " + error.message, "system");
    }
});

// ===== Generate ERD =====
async function generateERD() {
    let res = await fetch("/generate_erd", { method: "POST" });
    let data = await res.json();

    if (data.erd_img_url) {
        setOutput(`<img src="${data.erd_img_url}" style="max-width:100%;"/>`);
        addMessage("📊 ERD generated successfully.", "system");
    } else {
        addMessage("❌ Failed to generate ERD", "system");
    }
}

// ===== Generate Schema =====
async function generateSchema() {
    let res = await fetch("/generate_schema", { method: "POST" });
    let data = await res.json();

    if (data.graph_html_url) {
        setOutput(
            `<iframe src="${data.graph_html_url}" style="width:100%;height:700px;border:none;"></iframe>`
        );
        addMessage("📜 Schema generated successfully.", "system");
    } else {
        addMessage("❌ Failed to generate schema", "system");
    }
}

// ===== Inject Nodes =====
async function injectNodes() {
    let res = await fetch("/inject_nodes", { method: "POST" });
    let data = await res.json();

    if (data.status === "nodes_injected") {
        addMessage("✅ Nodes injected successfully.", "system");
    } else {
        addMessage("❌ Failed to inject nodes", "system");
    }
}

// ===== Inject Relationships =====
async function injectRelationships() {
    let res = await fetch("/inject_relationships", { method: "POST" });
    let data = await res.json();

    if (data.status === "relationships_injected") {
        addMessage("✅ Relationships injected successfully.", "system");
    } else {
        addMessage("❌ Failed to inject relationships", "system");
    }
}

// ===== Query Graph (Q/A Pane) =====
async function queryGraph() {
    let question = document.getElementById("question").value;

    if (!question.trim()) return;

    // Print user message
    addMessage(question, "user");

    let res = await fetch("/query_graph", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
    });

    let data = await res.json();

    // Print system (answer)
    if (data.answer) {
        addMessage("💡 " + data.answer, "system");
    } else {
        addMessage("⚠️ No answer returned", "system");
    }
}

async function runRCA() {
    const question = document.getElementById("question").value;
    if (!question.trim()) return;

    addMessage(question, "user");

    const res = await fetch("/stream_rca", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
    });

    if (!res.ok) {
        addMessage("⚠️ Failed to start RCA", "system");
        return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let partial = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        partial += decoder.decode(value, { stream: true });
        const lines = partial.split("\n");
        partial = lines.pop();  // keep incomplete line

        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const step = JSON.parse(line.trim());
                if (step.type === "tool" && step.query) {
                    addMessage(`🔍 Querying... '${step.query}'`, "system");
                } else if (step.type === "ai" && step.content) {
                    addMessage(`🤖 AI - ${step.content}`, "system");
                } else if (step.type === "user" && step.content) {
                    addMessage(step.content, "user");
                } else if (step.type === "system" && step.content) {
                    addMessage(step.content, "system");
                }
            } catch (err) {
                console.error("Failed to parse line:", line);
            }
        }
    }

    // Process remaining partial line
    if (partial.trim()) {
        try {
            const step = JSON.parse(partial.trim());
            if (step.type === "tool" && step.query) {
                addMessage(`🔍 Querying... '${step.query}'`, "system");
            } else if (step.type === "ai" && step.content) {
                addMessage(`🤖 AI - ${step.content}`, "system");
            } else if (step.type === "user" && step.content) {
                addMessage(step.content, "user");
            } else if (step.type === "system" && step.content) {
                addMessage(step.content, "system");
            }
        } catch (err) {
            console.error("Failed to parse partial line:", partial);
        }
    }
}