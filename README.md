# AI Interview Assistant

> A local, production-ready **Retrieval Augmented Generation (RAG)** system that answers questions about a candidate's resume and job description.
> Built with **Endee** vector database, **sentence-transformers** embeddings, and **Ollama + Mistral** for fully local LLM inference — no cloud API keys required.

---

## Architecture

```
User Query
    │
    ▼
sentence-transformers          ← embeds the query (all-MiniLM-L6-v2)
    │
    ▼
Endee Vector DB  ──search──►  Top-3 relevant chunks
    │                              │
    ▼                              ▼
Ollama (Mistral)  ◄── prompt with context + question
    │
    ▼
Answer  ──►  Streamlit UI
```

---

## RAG Workflow

| Step | Description |
|------|-------------|
| **1. Ingest** | Raw `.txt` files in `data/` are split into overlapping word-level chunks |
| **2. Embed** | Each chunk is encoded with `all-MiniLM-L6-v2` (384-dim dense vector) |
| **3. Store** | Vectors + metadata are upserted into an Endee cosine-similarity index |
| **4. Retrieve** | User query is embedded and the top-3 closest chunks are fetched via Endee ANN search |
| **5. Generate** | Retrieved chunks are packed into a prompt and sent to Ollama (Mistral) |
| **6. Answer** | Ollama returns a grounded, concise answer displayed in the Streamlit UI |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Database | **Endee** (Docker, local) |
| LLM | **Ollama** — `mistral` (fully local) |
| UI | **Streamlit** |

---

## Repository Structure

```
ai-interview-assistant/
├── app.py              # Streamlit web interface
├── ingest.py           # Document chunking, embedding, and Endee ingestion
├── rag_pipeline.py     # Core RAG logic: retrieval (Endee) + generation (Ollama)
├── config.py           # Centralised configuration and environment variables
├── requirements.txt    # Python dependencies
├── .env                # Local overrides (optional, safe defaults included)
├── README.md
└── data/
    ├── resume.txt          # Candidate resume
    └── job_description.txt # Target job description
```

---

## Prerequisites

- **Python 3.9+**
- **Docker** (for Endee vector database)
- **Ollama** ([install](https://ollama.com/download))

---

## Setup & Running

### 1. Clone the repository

```bash
git clone <repository_url>
cd ai-interview-assistant
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Endee Vector Database

```bash
docker run -d -p 8000:8080 endeeio/endee-server:latest
```

### 4. Start Ollama and pull Mistral

```bash
ollama serve          # starts the Ollama server
ollama pull mistral   # downloads the Mistral model (~4 GB)
```

### 5. (Optional) Customise your data

Replace or edit the files in `data/`:
- `data/resume.txt` — candidate's resume
- `data/job_description.txt` — target job description

### 6. Ingest documents into Endee

```bash
python ingest.py
```

Expected output:
```
Loading embedding model...
Connecting to Endee at http://localhost:8000 ...
Creating index 'interview_assistant'...
  'job_description.txt' → 12 chunks
  'resume.txt' → 14 chunks
Generating embeddings for 26 chunks...
Inserting into Endee Vector Database...
✅ Ingestion complete! Inserted 26 chunks into 'interview_assistant'.
```

### 7. Launch the UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Queries

- *Does the candidate have production RAG experience?*
- *What vector databases has the candidate worked with?*
- *Does the candidate meet the job's required qualifications?*
- *Summarise the skill gap between the candidate and the role.*
- *What is the compensation for this role?*

---

## Configuration

All settings have sensible defaults. Override via `.env`:

```env
ENDEE_BASE_URL=http://localhost:8000
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

To use a different Ollama model (e.g. `llama3`), pull it first:
```bash
ollama pull llama3
```
Then update `.env`:
```env
OLLAMA_MODEL=llama3
```

---

## How Endee Is Used

- **Index creation**: `client.create_index(name, dimension=384, space_type="cosine")`
- **Ingestion**: `index.upsert([{id, vector, meta: {document, source, chunk_id}}])`
- **Retrieval**: `index.query(vector=query_embedding, top_k=3)`

Endee's HNSW-based ANN search returns results in milliseconds even as the document corpus grows.

---

## License

MIT
