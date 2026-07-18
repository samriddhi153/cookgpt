# 🍳 CookGPT – Multi-Agent AI Cooking Assistant

CookGPT is a research-level AI cooking assistant built using LangGraph, LangChain, and multi-agent architecture.

## 🚀 Features
- Multi-Agent System (Chef, Nutritionist, Validator)
- LangGraph Workflow
- RAG (Retrieval-Augmented Generation)
- Multi-Model LLM (Groq + HuggingFace fallback)
- FastAPI Backend + Streamlit UI

## 📂 Project Structure
```
backend/
rag/
app/
config/
```

## ⚙️ Setup

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Add API keys
Rename `.env.example` to `.env` and fill keys.

### 3. Run backend
```
uvicorn backend.main:app --reload
```

### 4. Run UI
```
streamlit run app/streamlit_app.py
```

## 🔄 Workflow
User → Chef Agent → Nutritionist → Validator → Loop/Output

## 🧠 Tech Stack
- LangGraph
- LangChain
- Groq API
- HuggingFace
- FAISS

## 📌 Example Query
"Generate a low calorie paneer recipe"

---


