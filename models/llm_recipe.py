import ollama

class RecipeRecommender:
    def __init__(self, model="Mistral"):
        """ Initialize LLM for recipe recommendation """
        self.model = model

    def format_recipe_recommendation(self, recipe_list, user_query):
        """ Use LLM to generate a structured recipe recommendation """
        recipe_texts = "\n\n".join(
            [f"Recipe: {r['name']}\nIngredients: {r['ingredients']}\nInstructions: {r['steps']}" for r in recipe_list]
        )

        prompt = f"""
        The user asked: "{user_query}"
        
        Based on the user's request, identify if they are looking for a general, muscle-building, or weight-loss meal.
        If they are looking for muscle-building or weight-loss, prioritize suitable recipes.
        
        Here are the top matching recipes:
        {recipe_texts}

        Generate a well-structured and readable response:
        - Start with a friendly introduction.
        - Clearly list the best recipe(s) based on their goal.
        - Summarize the main ingredients and steps.
        - If any ingredients are unsuitable for their goal, add a brief recommendation.

        The response should be concise, clear, and engaging.
        """

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

# Example usage
if __name__ == "__main__":
    recommender = RecipeRecommender()
    
    # Mock recipe data (from recipe_retrieval.py)
    test_recipes = [
        {"name": "Grilled Chicken Salad", "ingredients": "chicken, lettuce, tomatoes, olive oil", "steps": "Grill chicken, chop vegetables, mix with dressing."},
        {"name": "Chicken and Brown Rice Bowl", "ingredients": "chicken, brown rice, vegetables", "steps": "Cook rice, stir-fry chicken with vegetables."}
    ]
    
    user_query = "I am trying to lose weight, what can I cook with chicken and rice?"
    formatted_output = recommender.format_recipe_recommendation(test_recipes, user_query)
    print(formatted_output)
