from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from rag.retriever import initialize_rag
from backend.graph.workflow import graph


class RequestModel(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[RAG] Initializing system...")
    initialize_rag()
    print("[RAG] System ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
def generate(request: RequestModel):
    result = graph.invoke({
        "user_input": request.query,
        "recipe": "",
        "nutrition": {},
        "is_valid": False,
        "retry_count": 0,
        "feedback": ""
    })

    return {
        "recipe": result["recipe"],
        "nutrition": result["nutrition"]
    }