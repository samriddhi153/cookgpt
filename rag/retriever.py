import os
import json
import pickle
import numpy as np
import faiss

# Lazy model loading
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[RAG] Loading embedding model: {model_name}...")
        _model = SentenceTransformer(model_name, device='cpu')
        print("[RAG] Model ready")
    return _model

# Config
DATA_PATH = "data/processed"
RECIPE_FILES = ["recipe_index.jsonl", "train.jsonl", "test.jsonl", "val.jsonl"]
INDEX_FILE = "rag_index.pkl"

# Global state
documents = []
metadata = []
index = None

# -----------------------------------
# LOAD ONLY RECIPE FILES
# -----------------------------------
def _load_recipe_documents():
    docs, meta = [], []
    for fname in RECIPE_FILES:
        path = os.path.join(DATA_PATH, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    title = item.get('title', '')
                    ingredients = item.get('ingredients', item.get('ingredients_list', ''))
                    instructions = item.get('instructions', item.get('steps', item.get('target', '')))
                    text = f"Recipe: {title}\nIngredients: {ingredients}\nInstructions: {instructions}"
                    docs.append(text)
                    meta.append({"source": fname, "title": title})
                except Exception:
                    continue
    return docs, meta

# -----------------------------------
# BUILD & SAVE INDEX
# -----------------------------------
def build_index():
    global index, documents, metadata
    print("[RAG] Building FAISS index...")
    model = _get_model()
    embeddings = model.encode(documents, batch_size=4)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    print(f"[RAG] Index built: {index.ntotal} recipes")

    # Save to disk
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({
            "documents": documents,
            "metadata": metadata,
            "index": index,
            "model_name": "all-MiniLM-L6-v2"
        }, f)
    print(f"[RAG] Saved to {INDEX_FILE}")

# -----------------------------------
# LOAD PERSISTED INDEX
# -----------------------------------
def load_index():
    global index, documents, metadata
    if os.path.exists(INDEX_FILE):
        print("[RAG] Loading existing index...")
        with open(INDEX_FILE, "rb") as f:
            data = pickle.load(f)
        documents = data["documents"]
        metadata = data["metadata"]
        index = data["index"]
        print(f"[RAG] Index loaded: {index.ntotal} recipes from cache")
        return True
    return False

# -----------------------------------
# INITIALIZE RAG
# -----------------------------------
def initialize_rag():
    global documents, metadata, index

    # Try loading from disk first
    if load_index():
        return

    # Build from scratch
    documents, metadata = _load_recipe_documents()
    if not documents:
        raise Exception("No recipe documents found in data/processed")
    build_index()

# -----------------------------------
# RETRIEVE
# -----------------------------------
def retrieve_context(query: str, k: int = 3):
    global index
    if index is None:
        raise Exception("Index not loaded. Call initialize_rag() first.")

    model = _get_model()
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            results.append(documents[i])
    return "\n\n".join(results)
