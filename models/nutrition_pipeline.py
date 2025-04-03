from models.llm_extraction import FoodExtractor
from models.usda_api import USDAFoodLookup

class NutritionPipeline:
    def __init__(self, api_key="YOUR_API_KEY", method="llm"):
        """ Initialize the pipeline with the chosen extraction method """
        self.api_key = api_key
        self.method = method  # Choose between "llm" or "ner" (future)
        self.usda = USDAFoodLookup(api_key)

        # Use LLM by default, but can switch to NER in the future
        if method == "llm":
            self.extractor = FoodExtractor()
#        elif method == "ner"：


    def process_text(self, text):
        extracted_foods = self.extractor.extract_food_with_llm(text)  # or extract_food_with_ner()
        nutrition_data = []

        for food_item in extracted_foods:
            food_name = food_item["food"]
            quantity = food_item["quantity"]


            # Query USDA API
            nutrition_info = self.usda.search_food(food_name)

            # Check the API response
            #print(f"USDA Response for {food_name}: {nutrition_info}")

            if nutrition_info:
                nutrition_info["quantity"] = quantity
                for key in ["calories", "protein", "fat", "carbs"]:
                    if key in nutrition_info:
                        nutrition_info[key] *= quantity
                nutrition_data.append(nutrition_info)

        return self.summarize_nutrition(nutrition_data)


    def summarize_nutrition(self, nutrition_data):
        """ Summarize total nutritional intake """
        total_calories = sum(item["calories"] for item in nutrition_data)
        total_protein = sum(item["protein"] for item in nutrition_data)
        total_fat = sum(item["fat"] for item in nutrition_data)
        total_carbs = sum(item["carbs"] for item in nutrition_data)

        return {
            "detailed_breakdown": nutrition_data,
            "summary": {
                "total_calories": total_calories,
                "total_protein": total_protein,
                "total_fat": total_fat,
                "total_carbs": total_carbs
            }
        }

# Example usage
if __name__ == "__main__":
    pipeline = NutritionPipeline(api_key="lmUwOuvbvhrvbfAtCzxt8M4ErPpJDbnTvQ1mZqij"  , method="llm")

    user_input = """
    I am so happy today. I went out and met two friends. Then I bought a pair of shoes.
    For breakfast, I had two eggs, a glass of orange juice, and some toast.
    Later, I had a bowl of rice with chicken for lunch.
    """
    
    result = pipeline.process_text(user_input)
    print(f"Final Nutrition Summary: {result}")
