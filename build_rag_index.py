#!/usr/bin/env python
"""Build and save RAG index once (run this separately)"""
import sys, os, json, pickle
sys.path.insert(0, '.')

# Minimal memory settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer
import numpy as np, faiss

# Only load recipe files (exclude user profiles)
RECIPE_FILES = ["recipe_index.jsonl", "train.jsonl", "test.jsonl", "val.jsonl"]
DATA_PATH = "data/processed"

documents = []
metadata = []

print("Loading recipes...")
for fname in RECIPE_FILES:
    path = os.path.join(DATA_PATH, fname)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                title = item.get('title', '')
                ingredients = item.get('ingredients', item.get('ingredients_list', ''))
                instructions = item.get('instructions', item.get('steps', item.get('target', '')))
                text = f"Recipe: {title}\nIngredients: {ingredients}\nInstructions: {instructions}"
                documents.append(text)
                metadata.append({"source": fname, "title": title})

print(f"[OK] {len(documents)} recipes loaded")

print("Loading embedding model...")
model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
model = SentenceTransformer(model_name, device='cpu')
print(f"[OK] Model ready (dim={model.get_sentence_embedding_dimension()})")

print("Building FAISS index (this may take a minute)...")
emb = model.encode(documents, batch_size=4, show_progress_bar=True)
dim = emb.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(emb))

# Save everything
with open("rag_index.pkl", "wb") as f:
    pickle.dump({
        "documents": documents,
        "metadata": metadata,
        "index": index,
        "model_name": "all-MiniLM-L6-v2"
    }, f)

print(f"[OK] Index saved to rag_index.pkl ({len(documents)} vectors)")
print("\nNext: run the app - it will load this index instantly.")
