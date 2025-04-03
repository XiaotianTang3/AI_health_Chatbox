import streamlit as st
from models.faq_retrieval import FAQRetriever
from models.nutrition_pipeline import NutritionPipeline
from models.recipe_retrieval import RecipeRetriever
from models.llm_recipe import RecipeRecommender

# Initialize FAQ retrieval, nutrition analysis, and recipe recommendation
faq_files = [
    "data/kidshealth_for_kids_faq.csv",
    "data/kidshealth_for_parents_faq.csv",
    "data/kidshealth_for_teens_faq.csv",
    "data/nutrition_faq.csv"
]
faq_retriever = FAQRetriever(faq_files)
nutrition_pipeline = NutritionPipeline(api_key="lmUwOuvbvhrvbfAtCzxt8M4ErPpJDbnTvQ1mZqij")
recipe_retriever = RecipeRetriever()
recipe_recommender = RecipeRecommender(model="mistral")

# Streamlit UI
st.title("AI Health Assistant")

# Tabs for FAQ search, nutrition analysis, and recipe recommendation
tab1, tab2, tab3 = st.tabs(["📚 Health FAQ", "🍽️ Nutrition Analysis", "👨‍🍳 Recipe Suggestions"])


# 📚 Health FAQ Search
with tab1:
    st.subheader("🔍 Enter Your Health Question")
    user_query = st.text_input("Question:", placeholder="What should I eat to stay healthy?")
    
    if st.button("Search FAQ"):
        if user_query.strip():
            answer = faq_retriever.search_faq(user_query)
            st.success(f"💡 AI Answer: {answer}")
        else:
            st.warning("Please enter a question!")

# 🍽️ Nutrition Analysis
with tab2:
    st.subheader("🥗 Log Your Meal")
    user_meal = st.text_area("Enter your meal description:", placeholder="I ate two eggs and a glass of milk.")

    if st.button("Analyze Meal"):
        if user_meal.strip():
            nutrition_result = nutrition_pipeline.process_text(user_meal)
            summary = nutrition_result["summary"]

            # Display nutrition analysis results
            st.success("📊 Your Nutrition Intake:")
            st.write(f"**Total Calories**: {summary['total_calories']} kcal")
            st.write(f"**Protein**: {summary['total_protein']} g")
            st.write(f"**Fat**: {summary['total_fat']} g")
            st.write(f"**Carbohydrates**: {summary['total_carbs']} g")
        else:
            st.warning("Please enter your meal record!")

# 👨‍🍳 Recipe Suggestions
with tab3:
    st.subheader("🍳 Find Delicious Recipes Based on Your Ingredients")
    user_ingredients = st.text_area("Tell me what ingredients you have, and I'll find the best recipes for you!", placeholder="E.g., chicken, rice, bell peppers")
    
    if st.button("Get Recipe Suggestions"):
        if user_ingredients.strip():
            # Retrieve and recommend recipes
            recipes = recipe_retriever.search_recipes(user_ingredients)
            response = recipe_recommender.format_recipe_recommendation(recipes, user_ingredients)
            st.success("🍽️ Here are some tasty recipes you can make!")
            st.write(response)
        else:
            st.warning("Oops! Please enter some ingredients so I can find recipes for you.")
