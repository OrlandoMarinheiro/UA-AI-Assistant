import os
import shutil
import time
import json
import threading
import queue
import torch
import uuid
from typing import List, Dict
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from IndexDocs import PDFIndexer
from chainOfThoughtRAG import ChainOfThoughtRAG

# --- Model Initialization ---
print("[STATUS] Loading model and processor...")
model_name = "vidore/colqwen2.5-v0.2"
collection_name = "FAA_collection"  

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[STATUS] Using device: {device}")

BATCH_SIZE = 4

model = ColQwen2_5.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2_5_Processor.from_pretrained(model_name)
print("[STATUS] Model and processor loaded")

# --- RAG Bot Setup ---
try:
    try:
        bot = ChainOfThoughtRAG(model=model, processor=processor, collection_name=collection_name)
    except TypeError:
        bot = ChainOfThoughtRAG()
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    class MockBot:
        def chain_retrieve_and_reason(self, msg, progress_callback=None):
            return "Mock Answer (RAG)", []
        def answer_without_rag(self, msg):
            return "Mock Answer (No RAG)"
    bot = MockBot()

# --- Indexer Setup ---
try:
    indexer = PDFIndexer(model=model, processor=processor, collection_name=collection_name)
    HAS_INDEXER = True
except Exception as e:
    HAS_INDEXER = False

indexing_lock = threading.Lock()

# --- File System Setup ---
BASE_DOCS_DIR = "docs"
DOCS_DIR = os.path.join(os.path.dirname(__file__), BASE_DOCS_DIR)

os.makedirs(DOCS_DIR, exist_ok=True)

app = FastAPI()
app.mount("/docs", StaticFiles(directory=DOCS_DIR), name="docs")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Dictionary to track indexing state
indexing_states: Dict[str, Dict] = {}

SERVER_BOOT_ID = str(uuid.uuid4())

@app.get("/boot-id")
async def get_boot_id():
    return {"boot_id": SERVER_BOOT_ID}

def update_progress(filename, progress_pct):
    if filename in indexing_states:
        indexing_states[filename]["progress"] = progress_pct

def process_indexing(file_path: str, filename: str):
    if HAS_INDEXER:
        with indexing_lock:
            try:
                print(f"[{filename}] Indexing started...")
                indexing_states[filename] = {"status": "indexing", "progress": 0}
                
                # Check if file still exists
                if not os.path.exists(file_path):
                    print(f"[{filename}] File not found. Aborting indexing.")
                    if filename in indexing_states: del indexing_states[filename]
                    return

                indexer.index_pdf(
                    file_path, 
                    batch_size=BATCH_SIZE,
                    progress_callback=lambda pct: update_progress(filename, pct)
                )
                
                # Final check
                if not os.path.exists(file_path):
                    if filename in indexing_states: del indexing_states[filename]
                    return

                indexing_states[filename] = {"status": "done", "progress": 100}
                print(f"[{filename}] Indexing finished.")
            except Exception as e:
                print(f"[{filename}] Error: {e}")
                indexing_states[filename] = {"status": "error", "progress": 0}
    else:
        indexing_states[filename] = {"status": "error", "progress": 0}      

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get('/ask/stream')
async def ask_stream(message: str, use_rag: str = "true"):
    q = queue.Queue()
    is_rag = use_rag.lower() == 'true'

    def progress_cb(obj): q.put(obj)
    
    def worker():
        try:
            if is_rag:
                # Use RAG logic
                ans, _ = bot.chain_retrieve_and_reason(message, progress_callback=progress_cb)
                if ans is None: q.put({"type": "status", "subtype": "done"})
                else: q.put({"type": "status", "subtype": "done", "answer": ans})
            else:
                # Use Direct LLM logic
                q.put({"type": "status", "subtype": "finalizing"}) # Notify UI that we are generating
                ans = bot.answer_without_rag(message)
                q.put({"type": "status", "subtype": "done", "answer": ans})

        except Exception as e:
            q.put({"type": "error", "error": str(e)})
            
    threading.Thread(target=worker, daemon=True).start()
    
    def event_stream():
        while True:
            try:
                item = q.get(timeout=20)
                yield f"data: {json.dumps(item)}\n\n"
                if isinstance(item, dict) and (item.get('type') == 'error' or item.get('subtype') == 'done'): break
            except queue.Empty: yield ": keep-alive\n\n"
            
    return StreamingResponse(event_stream(), media_type='text/event-stream')

@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    uploaded_files_info = []
    for file in files:
        file_path = BASE_DOCS_DIR + "/" + file.filename

        print(f"Uploading file: {file_path}")
        indexing_states[file.filename] = {"status": "queued", "progress": 0}

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        background_tasks.add_task(process_indexing, file_path, file.filename)
        uploaded_files_info.append({"name": file.filename, "status": "queued"})
    
    return JSONResponse(content={"message": "Upload successful", "files": uploaded_files_info})

@app.get("/indexing-status")
async def get_indexing_status(filename = None):
    if filename:
        return indexing_states.get(filename, {"status": "unknown", "progress": 0})
    return indexing_states

@app.get("/files")
async def list_files():
    files_list = []
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            
            # Skip files currently processing
            state = indexing_states.get(filename)
            if state and state["status"] in ["queued", "uploading", "indexing"]:
                continue

            if os.path.isfile(file_path):
                stats = os.stat(file_path)
                size_mb = stats.st_size / (1024 * 1024)
                creation_time = time.strftime('%Y-%m-%d', time.localtime(stats.st_mtime))
                files_list.append({
                    "name": filename,
                    "size": f"{size_mb:.2f} MB",
                    "date": creation_time
                })
    return files_list

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(DOCS_DIR, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f"[STATUS] File {filename} not found for removal.")

    if filename in indexing_states:
        del indexing_states[filename]
    
    if HAS_INDEXER:
        indexer.delete_file(filename)
    
    return {"message": "File and data successfully removed"}

@app.delete("/indexing-status/{filename}")
async def clear_indexing_status(filename: str):
    if filename in indexing_states:
        del indexing_states[filename]
        return {"message": "State cleared"}
    raise HTTPException(status_code=404, detail="State not found")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)