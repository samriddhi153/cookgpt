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
    Extract ingredient lines from recipe text using section detection
    and pattern matching.
    """

    lines = recipe_text.split("\n")
    ingredients = []

    in_section = False

    for line in lines:
        clean_line = line.strip()
        l = clean_line.lower()

        # Section detection
        if "ingredient" in l:
            in_section = True
            continue
        if in_section and any(x in l for x in ["step", "instruction", "direction", "method"]):
            in_section = False

        if in_section and clean_line:
            # Match bullets, numbers, or just descriptive lines in the section
            if re.match(r"^[\d\-\*\.\u2022]", l) or len(l) > 2:
                ingredients.append(l)

    # Fallback: if no section found, look for unit-bearing lines anywhere
    if not ingredients:
        for line in lines:
            l = line.strip().lower()
            if any(unit in l for unit in [
                "cup", "tbsp", "tsp", "gram", "oz", "tablespoon", "teaspoon"
            ]):
                ingredients.append(l)

    # Unique and limited
    seen = set()
    result = []
    for i in ingredients:
        if i not in seen:
            result.append(i)
            seen.add(i)

    return result[:10]


# -----------------------------------
# CLEAN INGREDIENT NAME
# -----------------------------------
def clean_ingredient(text: str):
    """
    Convert '1 cup chopped onions' to 'onion'
    """

    # Blacklist of non-caloric or problematic items
    blacklist = ["salt", "water", "pepper", "parsley", "thyme", "ice", "vinegar"]
    if any(b in text.lower() for b in blacklist):
        return ""

    # remove numbers + fractions
    text = re.sub(r"[\d\/\.\u00BC-\u00BE\u2150-\u215E]+", "", text)

    units = [
        "cup", "cups", "tbsp", "tsp", "gram", "g", "kg",
        "ml", "oz", "tablespoon", "teaspoons", "teaspoon", "tablespoons",
        "pound", "lb", "can", "bottle", "pinch", "dash"
    ]

    # remove units
    for unit in units:
        text = re.sub(r"\b" + unit + r"s?\b", "", text)

    # remove extra words/adjectives
    adjectives = [
        "chopped", "minced", "sliced", "fresh", "large", "small", "medium",
        "diced", "peeled", "grated", "shredded", "dried", "ground", "crushed",
        "negligible", "optional", "to taste", "low-sodium", "low-calorie"
    ]

    for adj in adjectives:
        text = re.sub(r"\b" + adj + r"\b", "", text)

    # remove symbols
    text = re.sub(r"[^a-zA-Z ]", "", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


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
    recipe -> ingredients -> USDA -> aggregated nutrition
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