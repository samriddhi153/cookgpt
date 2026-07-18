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
    feedback: str


# -------------------------
# CHEF AGENT
# -------------------------
def chef_agent(state: GraphState):
    user_input = state["user_input"]
    feedback = state.get("feedback", "")

    # RAG context
    try:
        context = retrieve_context(user_input)
    except Exception:
        context = "No external context available."

    feedback_prompt = ""
    if feedback:
        feedback_prompt = f"\n\nNOTE: Your previous attempt was rejected for these reasons: {feedback}. Please fix them."

    prompt = f"""
You are an expert AI Chef. Your goal is to provide high-quality, delicious, and healthy recipes.

CONTEXT FROM RECIPE DATABASE:
{context}

USER REQUEST & PREFERENCES:
{user_input}
{feedback_prompt}

Please follow these STRICT guidelines:
1. If Diet, Calories, or Cuisine preferences are provided above, ensure the recipe strictly adheres to them.
2. Structure your response clearly with two main sections: "Ingredients" and "Steps".
3. Use markdown for better readability.
4. Be encouraging and helpful.
"""

    recipe = llm.generate(prompt)

    return {
        **state,
        "recipe": recipe,
        "feedback": "" # Clear feedback after use
    }


# -------------------------
# VALIDATOR AGENT
# -------------------------
def validator_agent(state: GraphState):
    recipe = state["recipe"]

    is_valid = validate_recipe(recipe)

    feedback = ""
    if not is_valid:
        feedback = "Recipe must explicitly include 'Ingredients' and 'Steps' or 'Instructions' sections."

    return {
        **state,
        "is_valid": is_valid,
        "feedback": feedback
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
        return "nutrition"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print("[Workflow] Max retries reached")
        return "nutrition"

    return "retry"


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
        "retry": "retry",
        "nutrition": "nutrition"
    }
)

builder.add_edge("nutrition", END)
builder.add_edge("retry", "chef")

graph = builder.compile()