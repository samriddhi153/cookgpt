def validate_recipe(recipe: str) -> bool:
    """
    Basic validation:
    - Must contain Ingredients
    - Must contain Steps or Instructions
    """

    if not recipe:
        return False

    recipe_lower = recipe.lower()

    if "ingredient" in recipe_lower and (
        "step" in recipe_lower or "instruction" in recipe_lower
    ):
        return True

    return False