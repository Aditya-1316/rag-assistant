# 🛍️ RAG AI Shopping Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** shopping assistant that answers product questions using semantic vector search over an Amazon electronics catalog, powered by **FastAPI**, **Qdrant**, **Google Gemini**, and **Streamlit**.

---

## 🌐 Live Demo

- **Web App (Streamlit UI):** `https://YOUR_STREAMLIT_APP_URL.streamlit.app` *(replace with your Streamlit Cloud URL)*
- **API Docs (Swagger UI):** [https://rag-api-iezs.onrender.com/docs](https://rag-api-iezs.onrender.com/docs)

---

## 🏗️ Architecture

```text
User Query (Streamlit UI)
        │
        ▼
FastAPI Backend (/rag/)
        │
        ├──▶ 1. Generate query embedding (Gemini: gemini-embedding-001)
        ├──▶ 2. Vector similarity search (Qdrant: Amazon-items-collection-00)
        └──▶ 3. Grounded answer generation (Gemini: gemini-3.5-flash-lite w/ fallback)
```

### Tech Stack
- **Frontend:** Streamlit (`apps/chatbot_ui`)
- **Backend:** FastAPI & Uvicorn (`apps/api`)
- **Vector Database:** Qdrant / Qdrant Cloud (3072-dim Cosine similarity)
- **LLM & Embeddings:** Google GenAI SDK (`google-genai`)
- **Package & Container Management:** `uv` workspace + Docker & Docker Compose

---

## 📁 Project Structure

```text
.
├── apps/
│   ├── api/                    # FastAPI backend & RAG agent
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── src/
│   │       ├── agents/         # Retrieval & generation pipeline
│   │       └── api/            # FastAPI app, middleware, models & endpoints
│   └── chatbot_ui/             # Streamlit web frontend
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── src/chatbot_ui/
├── week_1/                     # Data exploration, preprocessing & RAG notebooks
├── docker-compose.yml
├── Makefile
├── pyproject.toml              # uv workspace root
└── uv.lock
```

---

## 🚀 Running Locally

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A **Google Gemini API key** (and optionally Qdrant Cloud credentials)

### 1. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
# Optional (defaults to local Docker Qdrant if omitted):
# QDRANT_URL=https://your-cluster.cloud.qdrant.io
# QDRANT_API_KEY=your_qdrant_api_key
```

### 2. Start with Docker Compose
```powershell
make run-docker-compose
```
*(Or run `docker compose up -d --build` directly.)*

Once running:
- **Streamlit UI:** `http://localhost:8501`
- **FastAPI Docs:** `http://localhost:8000/docs`
- **Qdrant Dashboard:** `http://localhost:6333/dashboard`

---

## ☁️ Cloud Deployment Stack

- **Vector Database:** [Qdrant Cloud](https://cloud.qdrant.io/) (Free Tier)
- **Backend API:** [Render](https://render.com/) (Docker deployment via `apps/api/Dockerfile`)
- **Frontend UI:** [Streamlit Community Cloud](https://share.streamlit.io/) (`apps/chatbot_ui/src/chatbot_ui/app.py`)

