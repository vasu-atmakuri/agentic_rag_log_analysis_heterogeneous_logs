"""Interactive service: HTTP upload with background ingestion, progress polling, and a WebSocket
chat that dispatches each question to the multi-type RAG pipeline or the tool-using agent.
The evaluation calls the same pipeline functions directly; this service is the user-facing shell.

Run: uvicorn log_analysis_system.interactive_log_analysis_service:app --port 8000
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .agentic_tool_use_loop import answer_tool_using_agent
from .local_language_model_backend import ModelRuntimeSettings, OllamaChatBackend
from .log_analysis_workspace import prepare_workspace
from .question_answering_methods import answer_multi_type_chunk_rag

MODEL = os.environ.get("LOG_ANALYSIS_MODEL", "qwen3:4b")
ARTIFACTS = os.environ.get("LOG_ANALYSIS_ARTIFACTS", os.path.join(tempfile.gettempdir(), "log_analysis_service"))

app = FastAPI(title="Log analysis service")
state = {"status": "idle", "workspace": None, "error": None, "filename": None}
backend = OllamaChatBackend(ModelRuntimeSettings(model=MODEL))

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Log analysis</title>
<style>body{font-family:system-ui;max-width:860px;margin:24px auto;padding:0 16px}#log{white-space:pre-wrap;border:1px solid #ccc;
padding:12px;min-height:240px}.tool{color:#666;font-size:.9em}</style></head><body>
<h2>Log analysis</h2><input type="file" id="f"><button onclick="up()">Upload</button> <span id="st"></span><br><br>
<select id="mode"><option value="agent">Agent</option><option value="rag">RAG</option></select>
<input id="q" size="70" placeholder="Ask about the log"><button onclick="ask()">Ask</button><div id="log"></div>
<script>
const ws=new WebSocket(`ws://${location.host}/chat`);const L=document.getElementById('log');
ws.onmessage=e=>{const m=JSON.parse(e.data);const d=document.createElement('div');if(m.type==='tool'){d.className='tool';
d.textContent=`[${m.tool}] ${JSON.stringify(m.arguments)}`}else{d.textContent=m.text}L.appendChild(d)};
async function up(){const fd=new FormData();fd.append('file',document.getElementById('f').files[0]);
await fetch('/upload',{method:'POST',body:fd});poll()}
async function poll(){const s=await (await fetch('/status')).json();document.getElementById('st').textContent=s.status;
if(s.status==='indexing')setTimeout(poll,1500)}
function ask(){ws.send(JSON.stringify({mode:document.getElementById('mode').value,question:document.getElementById('q').value}))}
</script></body></html>"""


def _ingest(path: str, name: str):
    try:
        state["workspace"] = prepare_workspace(name, path, ARTIFACTS, "automatic", backend=backend,
                                               characterization_model_tag=MODEL)
        state["status"] = "ready"
    except Exception as exc:  # noqa: BLE001
        state["status"], state["error"] = "failed", str(exc)


@app.get("/")
def index():
    return HTMLResponse(PAGE)


@app.post("/upload")
async def upload(file: UploadFile):
    os.makedirs(ARTIFACTS, exist_ok=True)
    name = os.path.splitext(os.path.basename(file.filename or "uploaded"))[0]
    path = os.path.join(ARTIFACTS, f"{name}.log")
    with open(path, "wb") as handle:
        handle.write(await file.read())
    state.update(status="indexing", workspace=None, error=None, filename=name)
    threading.Thread(target=_ingest, args=(path, name), daemon=True).start()
    return {"status": "indexing"}


@app.get("/status")
def status():
    return {"status": state["status"], "error": state["error"], "file": state["filename"]}


@app.websocket("/chat")
async def chat(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            message = await socket.receive_json()
            ws = state["workspace"]
            if ws is None:
                await socket.send_json({"type": "text", "text": "Upload a log and wait until it is ready."})
                continue
            fn = answer_tool_using_agent if message.get("mode") == "agent" else answer_multi_type_chunk_rag
            result = await asyncio.to_thread(fn, ws, backend, message.get("question", ""))
            for step in result.get("trace", []):
                if step["type"] == "tool":
                    await socket.send_json({"type": "tool", "tool": step["tool"], "arguments": step["arguments"]})
            await socket.send_json({"type": "text", "text": result["response_text"]})
    except WebSocketDisconnect:
        return
