import streamlit as st
import pandas as pd
from models.faq_retrieval import FAQRetriever
from models.nutrition_pipeline import NutritionPipeline
from models.recipe_retrieval import RecipeRetriever
from models.llm_recipe import RecipeRecommender
from streamlit.components.v1 import html

def add_fixed_input_css():
    st.markdown("""
    <style>
    /* This targets Streamlit's container divs */
    .stChatInputContainer {
        position: sticky !important;
        bottom: 0 !important;
        background-color: white !important;
        padding: 1rem !important;
        z-index: 999 !important; 
        width: 100% !important;
        background-color: var(--default-backgroundColor) !important;
    }
    
    .main .block-container {
        padding-bottom: 100px;
    }
    
    /* Adding a small arrow to indicate scrolling */
    .scroll-indicator {
        position: fixed;
        bottom: 80px;
        right: 20px;
        background-color: rgba(70, 70, 70, 0.5);
        color: white;
        padding: 10px;
        border-radius: 50%;
        cursor: pointer;
        z-index: 1000;
    }
    </style>
    
    <script>
    // Add a scroll to bottom button
    window.addEventListener('load', function() {
        const chatContainer = document.querySelector('.stChatMessageContent');
        if (chatContainer) {
            // Create scroll indicator
            const indicator = document.createElement('div');
            indicator.className = 'scroll-indicator';
            indicator.innerHTML = '↓';
            indicator.style.display = 'none';
            document.body.appendChild(indicator);
            
            // Scroll to bottom on load
            setTimeout(() => {
                window.scrollTo(0, document.body.scrollHeight);
            }, 500);
            
            // Show indicator when not at bottom
            window.addEventListener('scroll', function() {
                if ((window.innerHeight + window.scrollY) < document.body.scrollHeight - 100) {
                    indicator.style.display = 'block';
                } else {
                    indicator.style.display = 'none';
                }
            });
            
            // Scroll to bottom when indicator clicked
            indicator.addEventListener('click', function() {
                window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)

# Call this function at the start of your app
add_fixed_input_css()


def create_nutrition_dataframe(final_result):
    """
    Convert final_result["ingredients"] into a DataFrame and append unit suffixes.
    """
    items = final_result.get("ingredients", [])
    df = pd.DataFrame(items)
    if not df.empty:
        df.rename(columns={
            "food_name": "Food",
            "quantity": "Qty",
            "unit": "Unit",
            "calories": "Calories(kcal)",
            "protein": "Protein(g)",
            "fat": "Fat(g)",
            "carbs": "Carbs(g)"
        }, inplace=True, errors="ignore")
    return df

def format_summary_with_units(final_result):
    """
    Provide a summary of total nutrition with units:
    Calories (kcal), Protein (g), Fat (g), and Carbs (g).
    """
    if not final_result or final_result.get("type") != "dish":
        return "Unrecognized or not a dish."
    dish_name = final_result["dish_name"]
    tn = final_result["total_nutrition"]
    lines = [
        f"Dish Name: {dish_name}",
        "Total Nutrition:",
        f"- Calories: {tn['calories']} kcal",
        f"- Protein: {tn['protein']} g",
        f"- Fat: {tn['fat']} g",
        f"- Carbs: {tn['carbs']} g"
    ]
    return "\n".join(lines)

def strip_quantity(ing_str: str):
    """
    Keep only the food name by stripping away the quantity.
    e.g. "6 ounces penne" -> "penne"
    """
    import re
    pattern = re.compile(r'^([\d\/\.\s]+)([a-zA-Z]+)?\s+(.*)$')
    m = pattern.match(ing_str.strip())
    if m:
        return m.group(3).strip()
    else:
        return ing_str.strip()

def separate_main_sub(ingredients, pipeline):
    """
    Classify the ingredient list (which may include quantities) into main and secondary ingredients,
    and return only the food names (without quantities).
    """
    main_ings, sub_ings = pipeline.classify_ingredients(ingredients)
    main_names = [strip_quantity(m) for m in main_ings]
    sub_names = [strip_quantity(s) for s in sub_ings]
    return main_names, sub_names

def display_detected_items(items):
    """
    Display all detected food items.
    """
    items_text = "Detected the following food items:\n\n"
    for i, item in enumerate(items):
        if item["type"] == "dish":
            ingredients_str = "Main ingredients: " + ", ".join([strip_quantity(ing) for ing in item.get("ingredients", [])])
            items_text += f"{i+1}. {item['name']} (Dish)\n   {ingredients_str}\n"
        else:
            items_text += f"{i+1}. {item['name']} (Single Ingredient)\n"
    return items_text


# Initialize FAQ retrieval, nutrition analysis, and recipe recommendation
faq_files = [
    "data/kidshealth_for_kids_faq.csv",
    "data/kidshealth_for_parents_faq.csv",
    "data/kidshealth_for_teens_faq.csv",
    "data/nutrition_faq.csv"
]
faq_retriever = FAQRetriever(faq_files)
nutrition_pipeline = NutritionPipeline(api_key="lmUwOuvbvhrvbfAtCzxt8M4ErPpJDbnTvQ1mZqij", method='hybrid')
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
    st.subheader("Nutrition Analysis Chat")

    # Initialize state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "all_detected_items" not in st.session_state:
        st.session_state.all_detected_items = []
    if "awaiting_item_confirmation" not in st.session_state:
        st.session_state.awaiting_item_confirmation = False

    # Add JavaScript for auto-scrolling to the bottom
    st.markdown("""
    <script>
        // Scroll to the bottom after the page loads
        window.addEventListener('load', function() {
            setTimeout(function() {
                window.scrollTo(0, document.body.scrollHeight);
            }, 500);
        });

        // Observe DOM changes and scroll to the bottom when new messages appear
        const observer = new MutationObserver(function() {
            window.scrollTo(0, document.body.scrollHeight);
        });

        // Start the observer
        setTimeout(function() {
            const chatContainer = document.querySelector('.stChatMessageContainer');
            if (chatContainer) {
                observer.observe(chatContainer, { childList: true, subtree: true });
            }
        }, 1000);
    </script>
    """, unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        if role == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
            if "data" in msg and msg["data"] is not None:
                st.chat_message("assistant").dataframe(msg["data"])

    # User input
    user_input = st.chat_input("Please enter the food or drink you want to analyze (multiple items are allowed):")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # If waiting for the user to confirm the list of items
        if st.session_state.awaiting_item_confirmation:
            if user_input.lower() in ["analyze", "confirm", "ok", "yes", "start analysis"]:
                # Analyze all confirmed items
                st.session_state.chat_history.append({"role": "assistant", "content": "Starting nutrition analysis for all items..."})
                st.chat_message("assistant").write("Starting nutrition analysis for all items...")

                # Analyze each item and display results
                all_nutrition = {
                    "calories": 0,
                    "protein": 0,
                    "fat": 0,
                    "carbs": 0
                }

                for item in st.session_state.all_detected_items:
                    if item["type"] == "dish":
                        # Analyze a dish
                        dish_name = item["name"]
                        ingredients = item["ingredients"]

                        result = nutrition_pipeline._analyze_ingredients_list(ingredients, dish_name)
                        summary = format_summary_with_units(result)

                        st.session_state.chat_history.append({"role": "assistant", "content": summary})
                        st.chat_message("assistant").write(summary)

                        df = create_nutrition_dataframe(result)
                        if not df.empty:
                            st.session_state.chat_history.append({"role": "assistant", "content": "Detailed nutrition data:", "data": df})
                            st.chat_message("assistant").dataframe(df)

                        # Accumulate total nutrition
                        tn = result["total_nutrition"]
                        all_nutrition["calories"] += tn["calories"]
                        all_nutrition["protein"] += tn["protein"]
                        all_nutrition["fat"] += tn["fat"]
                        all_nutrition["carbs"] += tn["carbs"]

                    else:
                        # Analyze a single ingredient
                        food_name = item["name"]
                        quantity = item.get("quantity", 1.0)
                        nutrition = nutrition_pipeline.get_food_nutrition(food_name, quantity)

                        s = (f"[Single Item] {food_name}\n"
                             f"calories: {nutrition['calories']} kcal\n"
                             f"protein: {nutrition['protein']} g\n"
                             f"fat: {nutrition['fat']} g\n"
                             f"carbs: {nutrition['carbs']} g\n")

                        st.session_state.chat_history.append({"role": "assistant", "content": s})
                        st.chat_message("assistant").write(s)

                        # Accumulate total nutrition
                        all_nutrition["calories"] += nutrition["calories"]
                        all_nutrition["protein"] += nutrition["protein"]
                        all_nutrition["fat"] += nutrition["fat"]
                        all_nutrition["carbs"] += nutrition["carbs"]

                # Display total nutrition summary
                if len(st.session_state.all_detected_items) > 1:
                    total = (f"[Total Nutrition for All Items]\n"
                             f"calories: {round(all_nutrition['calories'], 2)} kcal\n"
                             f"protein: {round(all_nutrition['protein'], 2)} g\n"
                             f"fat: {round(all_nutrition['fat'], 2)} g\n"
                             f"carbs: {round(all_nutrition['carbs'], 2)} g\n")

                    st.session_state.chat_history.append({"role": "assistant", "content": total})
                    st.chat_message("assistant").write(total)

                # Reset state
                st.session_state.awaiting_item_confirmation = False
                st.session_state.all_detected_items = []

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "All items have been analyzed. You can enter new foods or dishes for analysis."
                })
                st.chat_message("assistant").write("All items have been analyzed. You can enter new foods or dishes for analysis.")

            elif user_input.lower().startswith("delete"):
                try:
                    # Delete the specified item by index
                    index = int(user_input[6:]) - 1
                    if 0 <= index < len(st.session_state.all_detected_items):
                        item = st.session_state.all_detected_items.pop(index)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Deleted: {item['name']}"
                        })
                        st.chat_message("assistant").write(f"Deleted: {item['name']}")

                        # Display the updated list
                        if st.session_state.all_detected_items:
                            updated_list = display_detected_items(st.session_state.all_detected_items)
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": updated_list
                            })
                            st.chat_message("assistant").write(updated_list)

                            prompt = "Please continue modifying or enter 'analyze' for nutrition calculation."
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": prompt
                            })
                            st.chat_message("assistant").write(prompt)
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": "All items have been deleted. Please enter new items for analysis."
                            })
                            st.chat_message("assistant").write("All items have been deleted. Please enter new items for analysis.")
                            st.session_state.awaiting_item_confirmation = False
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Invalid number, please try again."
                        })
                        st.chat_message("assistant").write("Invalid number, please try again.")
                except:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Invalid format, please use 'delete1' format."
                    })
                    st.chat_message("assistant").write("Invalid format, please use 'delete1' format.")

            elif user_input.lower().startswith("add"):
                # Add a new item
                new_food = user_input[3:].strip()
                if new_food:
                    st.session_state.all_detected_items.append({
                        "type": "food",
                        "name": new_food,
                        "quantity": 1.0
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Added: {new_food}"
                    })
                    st.chat_message("assistant").write(f"Added: {new_food}")

                    # Display the updated list
                    updated_list = display_detected_items(st.session_state.all_detected_items)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": updated_list
                    })
                    st.chat_message("assistant").write(updated_list)

                    prompt = "Please continue modifying or enter 'analyze' for nutrition calculation."
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": prompt
                    })
                    st.chat_message("assistant").write(prompt)
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Please specify the name of the item to add."
                    })
                    st.chat_message("assistant").write("Please specify the name of the item to add.")
            else:
                # Unrecognized command
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Unrecognized command. Please use 'add item name', 'delete number', or 'analyze'."
                })
                st.chat_message("assistant").write("Unrecognized command. Please use 'add item name', 'delete number', or 'analyze'.")

        else:
            # New analysis request
            result = nutrition_pipeline.process_text(user_input, use_standard_recipe=True)

            if not result:
                st.session_state.chat_history.append({"role": "assistant", "content": "Unable to understand your input. Please provide a clearer description of the food or dish."})
                st.chat_message("assistant").write("Unable to understand your input. Please provide a clearer description of the food or dish.")
            else:
                # Extract all dishes and foods
                all_items = []

                result_type = result.get("type", "")

                if result_type == "combined":
                    # Combined results: multiple dishes + other ingredients
                    dishes = result.get("dishes", [])
                    other_foods = result.get("other_foods", [])

                    # Add dishes
                    for dish in dishes:
                        all_items.append({
                            "type": "dish",
                            "name": dish["dish_name"],
                            "ingredients": dish["ingredients"]
                        })

                    # Add individual ingredients
                    for food in other_foods:
                        all_items.append({
                            "type": "food",
                            "name": food["food"],
                            "quantity": food.get("quantity", 1.0)
                        })

                elif result_type == "dish":
                    # Single dish
                    all_items.append({
                        "type": "dish",
                        "name": result["dish_name"],
                        "ingredients": result["ingredients"]
                    })

                elif result_type == "single":
                    # Single ingredient
                    item = result["items"][0]
                    all_items.append({
                        "type": "food",
                        "name": item["food_name"],
                        "quantity": 1.0
                    })

                elif result_type == "multiple":
                    # Multiple ingredients
                    for item in result["detailed_breakdown"]:
                        all_items.append({
                            "type": "food",
                            "name": item["food_name"],
                            "quantity": item.get("quantity", 1.0)
                        })

                # Save detected items and display them
                st.session_state.all_detected_items = all_items

                if all_items:
                    items_text = display_detected_items(all_items)
                    st.session_state.chat_history.append({"role": "assistant", "content": items_text})
                    st.chat_message("assistant").write(items_text)

                    prompt_txt = "Please confirm the list above. If you need to modify, enter the number to delete (e.g., 'delete1') or add a new item (e.g., 'add vegetable salad'). Once finished, enter 'analyze' to start the nutrition calculation."
                    st.session_state.chat_history.append({"role": "assistant", "content": prompt_txt})
                    st.chat_message("assistant").write(prompt_txt)

                    st.session_state.awaiting_item_confirmation = True
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "No food or dishes recognized."})
                    st.chat_message("assistant").write("No food or dishes recognized.")

    # Add a placeholder as an anchor at the bottom of the page
    bottom_placeholder = st.empty()

    # Force scrolling to the bottom after every update
    st.markdown("""
    <script>
        // Scroll to the bottom
        window.scrollTo(0, document.body.scrollHeight);
    </script>
    """, unsafe_allow_html=True)


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