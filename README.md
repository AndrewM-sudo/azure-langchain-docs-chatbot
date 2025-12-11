# azure-langchain-docs-chatbot

Use Python, LangChain, Azure OpenAI, FastAPI, and a React frontend, packaged in Docker containers.

***

## Prerequisites and tools

Have these ready before starting:

- Languages and runtimes  
  - Python 3.10+  
  - Node.js 18+ (for React)  

- Python packages (install in a virtualenv)  
  ```bash
  pip install "langchain>=0.3.0" \
              "langchain-openai>=0.2.0" \
              fastapi uvicorn[standard] \
              python-dotenv \
              chromadb \
              pypdf
  ```
  These cover LangChain, OpenAI/Azure integration, FastAPI, a local vector store (Chroma), and PDF loading.[1][2]

- Node/React tooling  
  ```bash
  npm create vite@latest frontend -- --template react-ts
  cd frontend
  npm install
  cd ..
  ```
  Vite + React + TypeScript gives a light, modern setup.[3]

- Azure setup  
  - Azure subscription  
  - Azure OpenAI (or Azure AI Foundry) with:
    - One chat model deployment (e.g., GPT-4o-like)  
    - One embeddings model deployment (e.g., a `text-embedding` model)  
  - Note:
    - Endpoint URL  
    - API key  
    - Deployment names for chat and embeddings[4][5]

- Dev tooling  
  - Git + GitHub account  
  - VS Code with Python and TypeScript extensions  
  - Docker Desktop (or other Docker runtime)[1][3]

***

## Step 1 – Create repo structure

```bash
mkdir azure-langchain-docs-bot
cd azure-langchain-docs-bot
git init
```

Create this structure:

```text
azure-langchain-docs-bot/
  backend/
    app/
      __init__.py
      config.py
      rag_chain.py
      main.py
    ingest/
      __init__.py
      ingest.py
    data/
      sample-docs/
        README.md
    vectorstore/
      .gitignore
    tests/
      __init__.py
  frontend/   # Vite React app (created earlier)
  .env.example
  docker-compose.yml
  backend.Dockerfile
  frontend.Dockerfile
  README.md
```

Add `backend/vectorstore/.gitignore`:

```text
*
!.gitignore
```

***

## Step 2 – Configure environment and Azure OpenAI

Fill `.env.example` at repo root:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_API_KEY="<your-api-key>"
AZURE_OPENAI_API_VERSION="2024-08-01-preview"

AZURE_OPENAI_CHAT_DEPLOYMENT="<your-chat-deployment-name>"
AZURE_OPENAI_EMBED_DEPLOYMENT="<your-embeddings-deployment-name>"

# Backend
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"

# CORS / Frontend
FRONTEND_ORIGIN="http://localhost:5173"
```

Copy to a real `.env`:

```bash
cp .env.example .env
```

Create `backend/app/config.py`:

```python
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(os.path.dirname(BASE_DIR), ".env")

load_dotenv(ENV_PATH)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "")

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
DATA_DIR = os.path.join(BASE_DIR, "data", "sample-docs")
```

***

## Step 3 – Set up ingestion pipeline (RAG indexing)

Create `backend/ingest/ingest.py` to load documents, split, embed with Azure embeddings via LangChain, and store in Chroma.[2][5]

```python
import os
from uuid import uuid4

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_DEPLOYMENT,
    DATA_DIR,
    VECTORSTORE_DIR,
)

def load_documents():
    docs = []

    # Text and markdown
    txt_md_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.[tm][dx][t]",  # .txt .md
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs.extend(txt_md_loader.load())

    # PDFs
    pdf_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs.extend(pdf_loader.load())

    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name="docs_collection",
        ids=[str(uuid4()) for _ in chunks],
    )

    vectorstore.persist()
    return vectorstore

def main():
    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print("Splitting documents ...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Building vectorstore ...")
    build_vectorstore(chunks)
    print(f"Vectorstore persisted to {VECTORSTORE_DIR}")

if __name__ == "__main__":
    main()
```

Add a sample doc, e.g. `backend/data/sample-docs/README.md` with any content you like (e.g., your own notes).[6]

Run the ingestion:

```bash
cd backend
python ingest/ingest.py
cd ..
```

***

## Step 4 – Implement RAG chain with LangChain

Create `backend/app/rag_chain.py`:

```python
from typing import Any, Dict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBED_DEPLOYMENT,
    VECTORSTORE_DIR,
)

def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0.1,
    )

def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name="docs_collection",
    )

def get_retriever():
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": 4})

def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()

    template = """
You are a helpful assistant answering questions based only on the provided context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say you don't know.
Provide concise, clear answers.
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip(),
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain

def answer_question(question: str) -> Dict[str, Any]:
    qa = get_qa_chain()
    result = qa.invoke({"query": question})
    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

    sources = []
    for d in source_docs:
        meta = d.metadata or {}
        sources.append(
            {
                "source": meta.get("source", ""),
                "page": meta.get("page", None),
                "snippet": d.page_content[:200],
            }
        )

    return {"answer": answer, "sources": sources}
```

Quick CLI test file `backend/app/cli_test.py`:

```python
from app.rag_chain import answer_question

def main():
    print("Ask a question (Ctrl+C to exit):")
    while True:
        q = input("> ")
        if not q.strip():
            continue
        result = answer_question(q)
        print("Answer:\n", result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print("-", s)

if __name__ == "__main__":
    main()
```

Run:

```bash
cd backend
python app/cli_test.py
cd ..
```

***

## Step 5 – Build FastAPI backend API

Create `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import FRONTEND_ORIGIN
from .rag_chain import answer_question

app = FastAPI(title="Azure LangChain Docs Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = answer_question(req.question)
    return result
```

Run backend locally:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
cd ..
```

Test with `curl`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project about?"}'
```

***

## Step 6 – Implement React frontend (Vite + TS)

Assuming you already ran `npm create vite@latest frontend -- --template react-ts` earlier.[3]

Update `frontend/vite.config.ts` to proxy API calls in dev:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

Replace `frontend/src/App.tsx`:

```tsx
import React, { useState } from "react";
import "./App.css";

type Source = {
  source: string;
  page?: number | null;
  snippet: string;
};

type ChatResponse = {
  answer: string;
  sources: Source[];
};

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }

      const data: ChatResponse = await res.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Docs Chat (Azure + LangChain)</h1>
        <p>Ask questions about your indexed documents.</p>
      </header>

      <main className="app-main">
        <form onSubmit={handleSubmit} className="chat-form">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your docs..."
            rows={3}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Thinking..." : "Ask"}
          </button>
        </form>

        {error && <div className="error">Error: {error}</div>}

        {answer && (
          <section className="answer-section">
            <h2>Answer</h2>
            <p>{answer}</p>
          </section>
        )}

        {sources.length > 0 && (
          <section className="sources-section">
            <h2>Sources</h2>
            <ul>
              {sources.map((s, idx) => (
                <li key={idx}>
                  <strong>{s.source}</strong>
                  {typeof s.page === "number" && ` (page ${s.page})`}
                  <br />
                  <small>{s.snippet}</small>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
```

Add minimal styling in `frontend/src/App.css`:

```css
.app {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 800px;
  margin: 0 auto;
  padding: 1.5rem;
}

.app-header {
  margin-bottom: 1.5rem;
}

.app-header h1 {
  margin: 0 0 0.5rem;
}

.chat-form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.chat-form textarea {
  resize: vertical;
  padding: 0.75rem;
  font-size: 1rem;
}

.chat-form button {
  align-self: flex-end;
  padding: 0.5rem 1.25rem;
  font-size: 0.95rem;
  cursor: pointer;
}

.answer-section,
.sources-section {
  margin-top: 1.5rem;
}

.error {
  color: #b00020;
  margin-top: 1rem;
}
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
cd ..
```

With backend running on port 8000 and the proxy config, React calls `/api/chat` which is forwarded to FastAPI `/chat`.

***

## Step 7 – (Optional) Upload-docs endpoint and UI

You can add an upload endpoint so users can add docs without rebuilding the image.[6][3]

Example minimal backend piece (append to `backend/app/main.py`):

```python
import os
from fastapi import UploadFile, File
from .config import DATA_DIR
from ..ingest.ingest import main as run_ingest

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    uploads_dir = os.path.join(DATA_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    dest_path = os.path.join(uploads_dir, file.filename)

    with open(dest_path, "wb") as f:
        f.write(await file.read())

    # Simple: re-run full ingest after each upload (ok for small demos)
    run_ingest()

    return {"status": "ok", "filename": file.filename}
```

React side: add a simple file input and POST to `/api/upload`. Keep in mind this re-ingests everything every time; for a real app, you’d index incrementally.[6]

***

## Step 8 – Containerization with Docker and docker-compose

### Backend Dockerfile

Create `backend.Dockerfile` at repo root:

```dockerfile
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY .env /app/.env

WORKDIR /app/backend

RUN python ingest/ingest.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `backend/requirements.txt`:

```text
langchain>=0.3.0
langchain-openai>=0.2.0
fastapi
uvicorn[standard]
python-dotenv
chromadb
pypdf
langchain-community
```

### Frontend Dockerfile

Create `frontend.Dockerfile` at repo root:

```dockerfile
FROM node:20 AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./frontend/
WORKDIR /app/frontend
RUN npm install

COPY frontend/ /app/frontend
RUN npm run build

FROM nginx:alpine AS prod
COPY --from=build /app/frontend/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### docker-compose

Create `docker-compose.yml`:

```yaml
version: "3.9"

services:
  backend:
    build:
      context: .
      dockerfile: backend.Dockerfile
    container_name: docs-bot-backend
    env_file:
      - .env
    ports:
      - "8000:8000"
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: frontend.Dockerfile
    container_name: docs-bot-frontend
    ports:
      - "5173:80"
    depends_on:
      - backend
    restart: unless-stopped
```

Build and run:

```bash
docker-compose build
docker-compose up
```

Access:

- Backend: `http://localhost:8000`  
- Frontend: `http://localhost:5173`

***

## Step 9 – Testing and GitHub polish

- Add simple tests in `backend/tests/` for:
  - `ingest.load_documents` and `split_documents`  
  - `answer_question` (mock the model to avoid live calls).[5][7]

- Add `README.md` explaining:
  - What the project does (RAG over docs using Azure OpenAI + LangChain)  
  - Setup steps (env vars, ingestion, running backend/frontend, docker-compose)  
  - Example `curl` command and screenshots of the React UI.[4][3]

- Push to GitHub with a clear name and description, plus maybe a small architecture diagram (mermaid or image).[1][3]

If you want, the next step can be a README-ready snippet describing the project’s purpose and architecture in 2–3 short sections.

[1](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/langchain)
[2](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview)
[3](https://www.praktikai.com/blog/2025/05/13/Setting-Up-a-RAG-Chatbot-on-Azure-AI-Foundry/)
[4](https://learn.microsoft.com/en-us/azure/ai-foundry/tutorials/copilot-sdk-build-rag?view=foundry-classic)
[5](https://learn.microsoft.com/en-us/azure/search/search-get-started-rag)
[6](https://uds.systems/blog/ai-assistant-knowledge-base-document-management/)
[7](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/rag-chatbot)
