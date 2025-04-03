import requests

class USDAFoodLookup:
    def __init__(self, api_key):
        """ Initialize USDA API with the provided key """
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    def search_food(self, food_name):
        """ Query USDA API for nutrition data of a food item """
        params = {
            "query": food_name,
            "api_key": self.api_key,
            "dataType": ["Foundation", "SR Legacy"],  # Use standard reference data
            "pageSize": 1  # Get only the best match
        }
        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "foods" in data and len(data["foods"]) > 0:
                return self._extract_nutrition_info(data["foods"][0])
        return None

    def _extract_nutrition_info(self, food_data):
        """ Extract relevant nutrition info from API response """
        nutrients = {}
        for nutrient in food_data["foodNutrients"]:
            if nutrient["nutrientName"] == "Energy":
                nutrients["calories"] = nutrient["value"]
            elif nutrient["nutrientName"] == "Protein":
                nutrients["protein"] = nutrient["value"]
            elif nutrient["nutrientName"] == "Total lipid (fat)":
                nutrients["fat"] = nutrient["value"]
            elif nutrient["nutrientName"] == "Carbohydrate, by difference":
                nutrients["carbs"] = nutrient["value"]

        return {
            "food": food_data["description"].lower(),
            "calories": nutrients.get("calories", 0),
            "protein": nutrients.get("protein", 0),
            "fat": nutrients.get("fat", 0),
            "carbs": nutrients.get("carbs", 0)
        }

# Example usage
if __name__ == "__main__":
    api_key = "lmUwOuvbvhrvbfAtCzxt8M4ErPpJDbnTvQ1mZqij"  
    usda = USDAFoodLookup(api_key)

    food = "sirloin steak"
    nutrition_info = usda.search_food(food)

    print(f"Nutrition info for {food}: {nutrition_info}")
