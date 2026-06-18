from typing import TypedDict
from langgraph.graph import StateGraph, END

# Import LLMService
from backend.services.llm_service import LLMService
from rag.retriever import retrieve_context
from backend.services.validation_service import validate_recipe
from backend.services.nutrition_service import get_nutrition

# -------------------------
# INIT LLM (GLOBAL SAFE)
# -------------------------
llm = LLMService()


# -------------------------
# STATE
# -------------------------
class GraphState(TypedDict):
    user_input: str
    recipe: str
    nutrition: dict
    is_valid: bool
    retry_count: int


# -------------------------
# CHEF AGENT
# -------------------------
def chef_agent(state: GraphState):
    user_input = state["user_input"]

    # RAG context
    try:
        context = retrieve_context(user_input)
    except Exception:
        context = "No external context available."

    prompt = f"""
You are an expert AI Chef.

Use this real recipe context:
{context}

User request:
{user_input}

Generate a detailed recipe with:
- Ingredients
- Steps
"""

    recipe = llm.generate(prompt)

    return {
        **state,
        "recipe": recipe
    }


# -------------------------
# VALIDATOR AGENT
# -------------------------
def validator_agent(state: GraphState):
    recipe = state["recipe"]

    is_valid = validate_recipe(recipe)

    return {
        **state,
        "is_valid": is_valid
    }


# -------------------------
# NUTRITION AGENT
# -------------------------
def nutrition_agent(state: GraphState):
    recipe = state["recipe"]

    nutrition = get_nutrition(recipe)

    return {
        **state,
        "nutrition": nutrition
    }


# -------------------------
# ROUTER (FIX LOOP)
# -------------------------
MAX_RETRIES = 2

def route_validation(state: GraphState):
    if state["is_valid"]:
        return END

    if state["retry_count"] >= MAX_RETRIES:
        print("[Workflow] Max retries reached")
        return END

    return "chef"


# -------------------------
# RETRY COUNTER NODE
# -------------------------
def increment_retry(state: GraphState):
    return {
        **state,
        "retry_count": state.get("retry_count", 0) + 1
    }


# -------------------------
# BUILD GRAPH
# -------------------------
builder = StateGraph(GraphState)

builder.add_node("chef", chef_agent)
builder.add_node("validator", validator_agent)
builder.add_node("nutrition", nutrition_agent)
builder.add_node("retry", increment_retry)

builder.set_entry_point("chef")

builder.add_edge("chef", "validator")

builder.add_conditional_edges(
    "validator",
    route_validation,
    {
        "chef": "retry",
        END: "nutrition"
    }
)

builder.add_edge("nutrition", END)
builder.add_edge("retry", "chef")

graph = builder.compile()