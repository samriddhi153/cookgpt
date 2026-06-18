import os
import requests
import re

USDA_API_KEY = os.getenv("USDA_API_KEY")
USDA_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


# -----------------------------------
# INGREDIENT EXTRACTION
# -----------------------------------
def extract_ingredients(recipe_text: str):
    """
    Extract ingredient lines from recipe text
    """

    lines = recipe_text.split("\n")
    ingredients = []

    for line in lines:
        line = line.strip().lower()

        # Look for ingredient-like lines
        if any(unit in line for unit in [
            "cup", "cups", "tbsp", "tsp", "gram", "g", "kg",
            "ml", "oz", "tablespoon", "teaspoon"
        ]):
            ingredients.append(line)

    return ingredients[:8]  # limit for API safety


# -----------------------------------
# CLEAN INGREDIENT NAME
# -----------------------------------
def clean_ingredient(text: str):
    """
    Convert '1 cup chopped onions' → 'onion'
    """

    # remove numbers + units
    text = re.sub(r"\d+", "", text)

    units = [
        "cup", "cups", "tbsp", "tsp", "gram", "g", "kg",
        "ml", "oz", "tablespoon", "teaspoon"
    ]

    for unit in units:
        text = text.replace(unit, "")

    # remove extra words
    text = re.sub(r"(chopped|minced|sliced|fresh|large|small)", "", text)

    # keep only letters
    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text.strip()


# -----------------------------------
# USDA QUERY
# -----------------------------------
def query_usda(ingredient: str):
    """
    Query USDA API for a single ingredient
    """

    if not USDA_API_KEY:
        print("[USDA] API key missing")
        return None

    params = {
        "api_key": USDA_API_KEY,
        "query": ingredient,
        "pageSize": 1
    }

    try:
        response = requests.get(USDA_URL, params=params, timeout=5)

        if response.status_code != 200:
            print(f"[USDA] Error: {response.status_code}")
            return None

        data = response.json()

        foods = data.get("foods", [])
        if not foods:
            return None

        food = foods[0]

        nutrients = food.get("foodNutrients", [])

        # Extract important nutrients
        nutrition = {
            "name": ingredient,
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbs": 0
        }

        for n in nutrients:
            name = n.get("nutrientName", "").lower()
            value = n.get("value", 0)

            if "energy" in name:
                nutrition["calories"] = value
            elif "protein" in name:
                nutrition["protein"] = value
            elif "fat" in name:
                nutrition["fat"] = value
            elif "carbohydrate" in name:
                nutrition["carbs"] = value

        return nutrition

    except Exception as e:
        print(f"[USDA] Exception: {e}")
        return None


# -----------------------------------
# MAIN FUNCTION
# -----------------------------------
def get_nutrition(recipe_text: str):
    """
    Full pipeline:
    recipe → ingredients → USDA → aggregated nutrition
    """

    ingredients_raw = extract_ingredients(recipe_text)

    if not ingredients_raw:
        print("[Nutrition] No ingredients extracted")
        return {}

    nutrition_results = []

    for item in ingredients_raw:
        cleaned = clean_ingredient(item)

        if not cleaned:
            continue

        data = query_usda(cleaned)

        if data:
            nutrition_results.append(data)

    # -----------------------------------
    # AGGREGATE TOTALS
    # -----------------------------------
    total = {
        "calories": 0,
        "protein": 0,
        "fat": 0,
        "carbs": 0
    }

    for item in nutrition_results:
        total["calories"] += item["calories"]
        total["protein"] += item["protein"]
        total["fat"] += item["fat"]
        total["carbs"] += item["carbs"]

    return {
        "ingredients_analyzed": nutrition_results,
        "total_nutrition": total
    }