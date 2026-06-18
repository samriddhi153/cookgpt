# CookGPT – Multi-Agent AI Cooking Assistant

<div align="center">

**CookGPT** is an AI-powered cooking assistant that generates personalized recipes using a multi-agent LangGraph workflow with Retrieval-Augmented Generation (RAG).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

CookGPT combines large language models with structured knowledge retrieval to create contextually relevant, nutritionally-aware recipes. The system uses a directed graph workflow where specialized agents collaborate to generate, validate, and analyze recipes.

---

## Data Sources

### Recipe Dataset

CookGPT uses the **Recipe1M+** dataset, one of the largest publicly available recipe collections:

- **Size**: Over 1 million recipes
- **Format**: JSONL (`data/processed/`)
- **Fields per recipe**:
  - `title`: Recipe name
  - `ingredients`: List of ingredients with quantities
  - `instructions`: Step-by-step cooking directions
  - `source`: Origin dataset (train/val/test splits)

The dataset is preprocessed into three files:
- `train.jsonl`: ~800K recipes for RAG indexing
- `val.jsonl`: ~100K recipes for validation
- `test.jsonl`: ~100K recipes for testing

Recipes are embedded using **Sentence Transformers** (`paraphrase-MiniLM-L3-v2`) and indexed with **FAISS** for efficient similarity search. The index (`rag_index.pkl`) is built once and reused across sessions.

### Nutrition Data

Nutritional information is retrieved live from the **USDA FoodData Central API**:
- Real-time calorie and macronutrient lookup
- Covers thousands of food ingredients
- Aggregated across extracted recipe ingredients

---

## How It Works

### Architecture

```
User Request (Streamlit UI)
         ↓
  POST /generate (FastAPI)
         ↓
   LangGraph State Machine
   ├── Chef Agent (LLM + RAG)
   ├── Validator Agent (schema check)
   ├── Nutrition Agent (USDA API)
   └── Loop on validation failure (max 2 retries)
         ↓
   JSON Response
         ↓
  Formatted UI Display
```

### Agents

**1. Chef Agent**
- Primary recipe generator
- Receives user query + top-k similar recipes from RAG as context
- Uses Groq LLaMA3-70B (fallback: HuggingFace DistilGPT2)
- Outputs structured recipe with ingredients and instructions

**2. Validator Agent**
- Checks recipe format compliance
- Ensures required sections (ingredients, instructions) exist
- Triggers regeneration loop if validation fails (max 2 retries)

**3. Nutrition Agent**
- Extracts ingredient lines from recipe text
- Cleans and normalizes ingredient names
- Queries USDA API for each ingredient
- Aggregates total calories, protein, fat, carbohydrates

### RAG Pipeline

1. **Index Build** (one-time): Load all `data/processed/*.jsonl` → encode with SentenceTransformer → FAISS L2 index → save to `rag_index.pkl`
2. **Retrieval**: For each query, encode query → FAISS search (k=3) → return top-k recipe texts as context
3. **Context Injection**: Retrieved recipes are appended to LLM prompt to guide generation

### State Machine

```python
State: {
    user_input: str,      # User query
    recipe: str,          # Generated recipe text
    nutrition: dict,      # USDA nutrition data
    is_valid: bool,       # Validation result
    retry_count: int      # Number of regeneration attempts
}
```

Flow: `Chef → Validator → (if valid) Nutrition → END` else `Chef ← Retry ← Validator`

---

## Project Structure

```
cookgpt/
├── backend/
│   ├── main.py                 # FastAPI application & endpoints
│   ├── graph/
│   │   └── workflow.py         # LangGraph state machine definition
│   └── services/
│       ├── llm_service.py      # LLM orchestration (Groq + HF fallback)
│       ├── nutrition_service.py # USDA API integration
│       └── validation_service.py # Recipe schema validator
├── app/
│   └── streamlit_app.py        # Streamlit user interface
├── rag/
│   ├── retriever.py            # FAISS + SentenceTransformer RAG
│   └── rag_index.pkl           # Generated recipe index (cached)
├── data/
│   ├── raw/                    # Original Recipe1M+ source files
│   └── processed/              # Curated JSONL (train/val/test splits)
├── requirements.txt            # Python dependencies
└── run.py                      # Launcher script (backend + UI)
```

---

## Key Components

### LLM Service (`backend/services/llm_service.py`)

Dual-provider setup for reliability:
- **Primary**: Groq API with `llama3-70b-8192` (fast, high-quality)
- **Fallback**: HuggingFace `distilgpt2` (local, no API needed)
- Auto-fallback on network/API errors

### Nutrition Service (`backend/services/nutrition_service.py`)

Ingredient extraction via pattern matching:
- Identifies quantity+unit lines (cup, tbsp, tsp, gram, oz, etc.)
- Cleans ingredient names (removes quantities, prep descriptors)
- Batch queries USDA API (up to 8 ingredients per recipe)
- Returns per-ingredient breakdown + aggregated totals

### RAG Retriever (`rag/retriever.py`)

- Lazy model loading (only initializes on first query)
- Disk-backed index persistence (loads `rag_index.pkl` if present)
- Batch encoding (batch_size=4) to manage memory
- Global document store for fast retrieval

### Validation Service (`backend/services/validation_service.py`)

Schema validation rules:
- Recipe must contain "ingredients" section
- Recipe must contain "instructions" or "steps" section
- Minimum length checks to filter empty/garbage outputs

---

## API Reference

### POST `/generate`

Generate a recipe from natural language.

**Request**:
```json
{
  "query": "low calorie vegan pasta with tofu"
}
```

**Response**:
```json
{
  "recipe": "## Creamy Vegan Pasta\n\n**Ingredients:**\n- 200g firm tofu...\n\n**Instructions:**\n1. Press tofu...",
  "nutrition": {
    "ingredients_analyzed": [
      {"name": "tofu", "calories": 144, "protein": 16, "fat": 8, "carbs": 2},
      {"name": "olive oil", "calories": 119, "protein": 0, "fat": 14, "carbs": 0}
    ],
    "total_nutrition": {
      "calories": 420,
      "protein": 18.5,
      "fat": 12.0,
      "carbs": 65.0
    }
  }
}
```

---

## Configuration

Environment variables (`.env`):

| Variable | Description | Required? |
|----------|-------------|-----------|
| `GROQ_API_KEY` | Groq API key (get from https://console.groq.com/keys) | Yes |
| `HUGGINGFACE_API_KEY` | HuggingFace token (optional, for HF fallback) | No |
| `USDA_API_KEY` | USDA FoodData Central API key | Yes |
| `EMBEDDING_MODEL` | SentenceTransformer model name | No (default: `paraphrase-MiniLM-L3-v2`) |

---

## Troubleshooting

### Backend hangs on startup (RAG)

**Cause**: SentenceTransformer downloading on first run.

**Fix**: Run `python -c "from rag.retriever import initialize_rag; initialize_rag()"` once to pre-download model & build index (~2–3 minutes).

### UnicodeEncodeError on Windows

**Cause**: Emoji/non-ASCII characters in console output.

**Fix**: All emoji have been removed from print statements in codebase.

### FAISS import error

**Fix**: `pip install faiss-cpu` (Windows/Linux) or `faiss-gpu` (CUDA).

### Out of memory during index build

**Fix**: Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in `.env`. Index build uses ~500MB RAM.

### USDA API returns 403/401

**Fix**: Verify USDA key is active at https://fdc.nal.usda.gov/api-key-signup.html. Free tier allows 1000 requests/day.

---

## Roadmap

- [ ] Add dietary restriction filtering (allergens, gluten-free, etc.)
- [ ] Support image input (food recognition from photo)
- [ ] Multi-language recipe generation
- [ ] Voice interface (text-to-speech / speech-to-text)
- [ ] Docker containerization
- [ ] CI/CD pipelines for automated testing

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Made with 🍳 by the CookGPT Team**
