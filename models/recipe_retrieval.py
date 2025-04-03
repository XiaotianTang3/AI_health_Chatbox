import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class RecipeRetriever:
    def __init__(self, recipe_file="data/recipe.csv", model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir="cache"):
        """ Initialize the recipe retriever with a cached ingredient embedding system. """
        self.model = SentenceTransformer(model_name)
        self.recipe_file = recipe_file
        self.cache_dir = cache_dir
        self.embeddings_file = os.path.join(cache_dir, "recipe_embeddings.pkl")

        # Load or encode ingredients
        if os.path.exists(self.embeddings_file):
            self.load_embeddings()
        else:
            self.build_embeddings()

    def load_embeddings(self):
        """ Load recipe embeddings from cache. """
        with open(self.embeddings_file, "rb") as f:
            self.recipes, self.ingredient_embeddings = pickle.load(f)

    def build_embeddings(self):
        """ Encode ingredients and save to cache. """
        df = pd.read_csv(self.recipe_file)
        
        self.recipes = df[["name", "ingredients", "steps"]].to_dict(orient="records")

        # Encode ingredients into embeddings
        ingredient_texts = [r["ingredients"] for r in self.recipes]
        self.ingredient_embeddings = self.model.encode(ingredient_texts, convert_to_numpy=True)

        # Save embeddings to pickle
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.embeddings_file, "wb") as f:
            pickle.dump((self.recipes, self.ingredient_embeddings), f)

    def search_recipes(self, user_input, top_k=7):
        """ Search for the most relevant recipes based on user input using cosine similarity. """
        query_embedding = self.model.encode([user_input], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.ingredient_embeddings)[0]

        # Get top-k matching recipes
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.recipes[i] for i in top_indices]

# Example usage
if __name__ == "__main__":
    retriever = RecipeRetriever()
    user_query = "I have beef and noodle, what can I cook?"
    results = retriever.search_recipes(user_query)

    for r in results:
        print(f"Recipe: {r['name']}\nIngredients: {r['ingredients']}\nInstructions: {r['steps']}\n")
