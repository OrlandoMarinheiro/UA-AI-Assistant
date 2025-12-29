# FAA Project – UA AI Assistant

This project implements a conversational AI chatbot designed to answer questions based on a closed knowledge base related to the *Fundamentals of Machine Learning* course.  
It combines **Retrieval-Augmented Generation (RAG)** with **Chain-of-Thought Augmented Generation (CAG)** and supports **multimodal documents** using ColPali and Qdrant.

---

## Project Structure

```
app.py
CAG_flow.drawio
chainOfThoughtRAG.py
docker-compose.yml
index.html
IndexDocs.py
README.md
requirements.txt
searcher.py
assets/
docs/                       --> folder where uploaded files are saved
qdrant_storage/
 ├── raft_state.json
 ├── aliases/
 │   └── data.json
 └── collections/
     └── FAA_collection/    --> indexed collection
```

---

## Prerequisites

- Python 3.10+
- Docker
- GPU recommended (for ColPali indexing)
- Groq API Key

---

## Installation & Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .env
source .env/Scripts/activate  
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want to use it for ColPali indexing, install the appropriate PyTorch packages:
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Docker

Download and install Docker from:  
https://www.docker.com/get-started/

---

## Qdrant Setup (Vector Database)

### 4. Pull the latest Qdrant image

```bash
docker pull qdrant/qdrant
```

### 5. Create and start the container

```bash
docker-compose up -d
```

### 6. (Optional) Start an existing container

```bash
docker-compose start
```

Qdrant will be available at:
- REST API: `http://localhost:6333`
- gRPC: `http://localhost:6334`

---

## Using an Existing Collection

If you already have a Qdrant collection:

1. After creating and starting the container, copy the collection folder into:

```
qdrant_storage/collections/
```

2. By default, the system stores uploaded documents in the `docs/` folder.  
If you manually import a collection into Qdrant, you must also create a `docs/` folder containing **the same documents** used to generate that collection.

This ensures consistency between stored embeddings and the original documents.


3. After doing these steps, you have to reboot the container
```bash
docker-compose restart qdrant
```
---
## Indexing a New Document
To index a document, simply select **My Knowledge Base** in the left sidebar, drag and drop the file(s), and click **Start Indexing**.  
The indexing queue will then start processing the documents.

---

## Obtain Groq API Key

Create a Groq API key at:
https://console.groq.com/keys

Define the API key as an environment variable:

```bash
export GROQ_API_KEY=YOUR_API_KEY
```

(On Windows PowerShell)
```powershell
$env:GROQ_API_KEY="YOUR_API_KEY"
```

---

## Running the Application

```bash
python app.py
```

Once running, the application will start the chatbot interface and allow document uploads, indexing, and querying.

---

## Indexing & Search

- **Indexing / ingestion**: see `IndexDocs.py`
- **Vector search & retrieval**: see `searcher.py`
- **RAG + Chain-of-Thought pipeline**: see `chainOfThoughtRAG.py`

---

## Notes

- `qdrant_storage/` contains local Qdrant data.  
  Do **not** commit this folder in production environments if it contains sensitive data.
- `docs/` must always match the documents used to build the current vector collection.
- Token usage depends on the limits imposed by the Groq API.
- Running ColPali search and indexing on CPU takes a lot of time, so it is strongly recommended to run it on a GPU.
- All the tests were done with an i5-13450HX, 16 GB RAM, and an RTX 4060 8GB VRAM.
- Depending on your VRAM, you may need to adjust the BATCH_SIZE in `app.py`. For 8GB of GPU memory, a BATCH_SIZE of 4–6 works well
---

## UA AI Assistant Demo
[Chatbot answer demo](https://youtu.be/qwpDaayPBAs)


## Authors

- **Nuno Miguel Paixão Loureiro** 
- **Orlando Martins Marinheiro** 
