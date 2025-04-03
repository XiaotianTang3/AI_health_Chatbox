import spacy
import json
import re
import sqlite3
import math  # Ensure the math library is imported
import ollama
from models.usda_api import USDAFoodLookup
from models.llm_extraction import FoodExtractor
from models.ner_extraction import NERFoodExtractor

class NutritionPipeline:
    def __init__(self, api_key="YOUR_API_KEY", method="hybrid"):
        self.api_key = api_key
        self.usda = USDAFoodLookup(api_key)
        self.method = method
        
        if method == "llm":
            self.llm_extractor = FoodExtractor()
            self.ner_extractor = None
        elif method == "ner":
            self.ner_extractor = NERFoodExtractor(model_path="food_ner_model")
            self.llm_extractor = None
        else:
            # Hybrid mode uses both LLM and NER extractors
            self.llm_extractor = FoodExtractor()
            self.ner_extractor = NERFoodExtractor(model_path="food_ner_model")

        print("Using SQLite database for recipe queries (no JSON fallback).")

        try:
            self.nlp = spacy.load("food_ner_model")
            print("Custom food NER model loaded.")
        except:
            print("Warning: custom NER model not found, falling back to en_core_web_sm.")
            self.nlp = spacy.load("en_core_web_sm")

        # Common dish keywords for simple identification
        self.common_dish_keywords = {
            "soup", "salad", "pizza", "burger", "sandwich", "stew", "curry",
            "cake", "pie", "pasta", "noodles", "omelet", "dumpling", "fried",
            "steak", "fries", "gratin", "casserole", "bake", "coke", "tea", "coffee"
        }
        # Simple keywords for main and secondary ingredients classification
        self.main_keywords = {"chicken", "beef", "pork", "fish", "egg", "shrimp", "rice", "noodles", "pasta", "milk", "tomato", "cheese", "penne", "turkey", "sausage", "coke", "coffee", "tea"}
        self.sub_keywords = {"salt", "pepper", "oil", "garlic", "onion", "butter", "sugar", "cream", "flour", "vinegar", "powder", "seasoning", "herbs", "spice", "sauce", "chilies"}
        
        # Standard nutritional information for common foods (per 100g/ml)
        self.standard_nutrition = {
            # Beverages
            "coke": {"calories": 37.5, "protein": 0, "fat": 0, "carbs": 10.6},
            "cola": {"calories": 37.5, "protein": 0, "fat": 0, "carbs": 10.6},
            "coca-cola": {"calories": 37.5, "protein": 0, "fat": 0, "carbs": 10.6},
            "pepsi": {"calories": 41, "protein": 0, "fat": 0, "carbs": 11},
            "ice": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0},
            "water": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0},
            
            # Dishes
            "mac and cheese": {"calories": 164, "protein": 8.5, "fat": 6.3, "carbs": 17.8},
            "macaroni and cheese": {"calories": 164, "protein": 8.5, "fat": 6.3, "carbs": 17.8},
            "mac & cheese": {"calories": 164, "protein": 8.5, "fat": 6.3, "carbs": 17.8},
            
            # Common ingredients
            "chicken": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
            "beef": {"calories": 250, "protein": 26, "fat": 17, "carbs": 0},
            "fish": {"calories": 136, "protein": 20, "fat": 5, "carbs": 0},
            "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
            "pasta": {"calories": 131, "protein": 5, "fat": 1.1, "carbs": 25},
            "bread": {"calories": 265, "protein": 9, "fat": 3.2, "carbs": 49}
        }
        
        # Conversion table for portion sizes (in ml or g)
        self.portion_sizes = {
            "cup": 240,        # ml
            "tablespoon": 15,  # ml
            "teaspoon": 5,     # ml
            "slice": 30,       # g
            "piece": 50,       # g
            "serving": 250     # g
        }

    def extract_dish_names(self, user_input: str):
        """
        Extract all possible dish or beverage names (FOOD entities) from the input text,
        and clean up any unwanted markers.
        """
        doc = self.nlp(user_input)
        ents = [ent.text.strip().lower() for ent in doc.ents if ent.label_ == "FOOD"]
        
        cleaned_ents = []
        if ents:
            for ent in ents:
                # Use the NERFoodExtractor's clean_entity_text method to clean the entity
                if hasattr(self.ner_extractor, 'clean_entity_text'):
                    cleaned = self.ner_extractor.clean_entity_text(ent)
                    if len(cleaned) >= 2:  # Ensure the entity is not empty
                        cleaned_ents.append(cleaned)
                else:
                    cleaned_ents.append(ent)
        
        # If no valid entities were found, attempt to extract food entities
        if not cleaned_ents:
            if self.ner_extractor is not None:
                extracted = self.ner_extractor.extract_food_with_ner(user_input)
                if extracted:
                    return [item["food"] for item in extracted]
            return [user_input.strip().lower()]
        
        return cleaned_ents

    def is_probably_dish(self, text: str) -> bool:
        """
        Determine if the given text is likely a dish or beverage.
        If it contains spaces or common keywords, it is considered a dish.
        """
        name = text.lower()
        if " " in name:
            return True
        for kw in self.common_dish_keywords:
            if kw in name:
                return True
        return False

    def search_recipe_in_db(self, dish_name: str):
        """
        Search the SQLite database for the dish_name.
        Returns (title, ingredients_list) if found, otherwise (None, None).
        The ingredients_list may include quantities.
        """
        try:
            conn = sqlite3.connect('recipes.db')
            cursor = conn.cursor()
            cursor.execute("SELECT title, ingredients FROM recipes WHERE title LIKE ?", ('%' + dish_name + '%',))
            row = cursor.fetchone()
            conn.close()
            if row:
                title, ingredients_json = row
                arr = json.loads(ingredients_json)
                final_list = []
                for item in arr:
                    if isinstance(item, dict) and "text" in item:
                        final_list.append(item["text"])
                    elif isinstance(item, str):
                        final_list.append(item)
                return title.lower(), final_list
            else:
                return None, None
        except Exception as e:
            print("DB error:", e)
            return None, None

    def generate_recipe_with_llm(self, dish_name: str):
        """
        Use an LLM to generate a list of ingredients for the dish.
        The output is a JSON array of strings with a maximum of 5-7 items.
        """
        prompt = f"""
        Provide a concise list of common ingredients with quantities for "{dish_name}".
        Only include the most commonly used ingredients for this dish.
        Format as a JSON array of strings, e.g. ["200g chicken", "1 onion"].
        Keep the list focused on main ingredients (maximum 5-7 items).
        For a simple item or beverage, only list its core components.
        For example, for "coke" or "cola", just list ["330ml Coca-Cola", "ice cubes"].
        """
        try:
            response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            text = response["message"]["content"]
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                arr = json.loads(match.group(0))
                # Limit the ingredient list to 7 items maximum
                return arr[:7]
            else:
                return []
        except Exception as e:
            print(f"LLM recipe generation error: {e}")
            return []

    def parse_ingredient(self, ingredient_str: str):
        """
        Parse an ingredient string (e.g. "200g chicken") into a tuple:
        (quantity, unit, food name)
        """
        pattern = re.compile(r'^([\d\/\.\s]+)([a-zA-Z]+)?\s+(.*)$')
        m = pattern.match(ingredient_str.strip())
        if m:
            qty_raw = m.group(1).strip()
            unit_raw = m.group(2) if m.group(2) else ""
            food_name = m.group(3).lower()
            total_qty = 0.0
            for part in qty_raw.split():
                try:
                    if "/" in part:
                        num, denom = part.split("/")
                        total_qty += float(num) / float(denom)
                    else:
                        total_qty += float(part)
                except:
                    total_qty += 1.0
            return total_qty, unit_raw.lower(), food_name.strip()
        else:
            return 1.0, "", ingredient_str.lower()

    def convert_to_grams(self, qty, unit):
        """Convert various units to grams."""
        unit = unit.lower()
        if unit in ["g", "gram", "grams"]:
            return qty
        elif unit in ["kg", "kilogram", "kilograms"]:
            return qty * 1000
        elif unit in ["oz", "ounce", "ounces"]:
            return qty * 28.35
        elif unit in ["lb", "pound", "pounds"]:
            return qty * 453.592
        elif unit in ["cup", "cups"]:
            return qty * 240  # More reasonable cup conversion
        elif unit in ["tbsp", "tablespoon", "tablespoons"]:
            return qty * 15  # Standard tablespoon
        elif unit in ["tsp", "teaspoon", "teaspoons"]:
            return qty * 5   # Standard teaspoon
        elif unit in ["ml", "milliliter"]:
            # Assume liquid density of 1g/ml
            return qty
        elif unit in ["piece", "pieces", "slice", "slices"]:
            # Use predefined portion sizes or default to 30g
            return qty * self.portion_sizes.get(unit.rstrip("s"), 30)
        return qty

    def _identify_food_category(self, food_name: str, calories_per_100g: float):
        """
        Identify the food category based on the food name and its calories per 100g.
        """
        food = food_name.lower()
        if any(word in food for word in ["chicken", "beef", "pork", "fish", "meat", "turkey"]):
            return "meat"
        elif any(word in food for word in ["rice", "pasta", "noodle", "bread", "cereal", "grain"]):
            return "grain"
        elif any(word in food for word in ["vegetable", "carrot", "broccoli", "spinach", "lettuce"]):
            return "vegetable"
        elif any(word in food for word in ["fruit", "apple", "orange", "banana", "berry"]):
            return "fruit"
        elif any(word in food for word in ["milk", "cheese", "yogurt", "cream", "dairy"]):
            return "dairy"
        elif any(word in food for word in ["oil", "butter", "margarine", "lard"]):
            return "oil"
        elif any(word in food for word in ["drink", "beverage", "water", "juice", "soda", "coke", "cola"]):
            return "beverage"
        if calories_per_100g > 800:
            return "oil"
        elif calories_per_100g > 300:
            return "meat"
        elif calories_per_100g > 200:
            return "grain"
        elif calories_per_100g < 30:
            return "vegetable"
        return "unknown"

    def _get_reasonable_max_factor(self, food_category: str):
        """
        Get a reasonable maximum factor for the given food category (relative to 100g/ml).
        """
        max_factors = {
            "meat": 5,       # 500g of meat is usually the upper limit
            "grain": 8,      # 800g of grains might be the upper limit
            "vegetable": 10, # 1kg of vegetables
            "fruit": 6,      # 600g of fruit
            "dairy": 8,      # 800ml of dairy products
            "oil": 2,        # 200g of oil
            "beverage": 15,  # 1.5L of beverage
            "unknown": 10    # Default upper limit
        }
        return max_factors.get(food_category, 10)

    def get_food_nutrition(self, food_name: str, amount=1.0, unit=""):
        """
        Calculate the nutritional values for a given food, intelligently processing
        the data from the API.
        """
        food = food_name.lower().strip()
        
        # Special handling for ice
        if food in ["ice", "ice cube", "ice cubes", "冰", "冰块"]:
            return {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
                
        if unit:
            grams = self.convert_to_grams(amount, unit)
        else:
            # Assume a standard serving of 100g if no unit is provided
            grams = amount * 100  
                
        factor = grams / 100.0
        
        # Check standard nutritional info first
        for key, std_nutr in self.standard_nutrition.items():
            if key == food or key in food or food in key:
                if "cola" in food or "coke" in food:
                    if unit == "":
                        factor = 3.3  # Assume one can (330ml) if unit is not provided
                    if factor > 10:
                        factor = min(factor, 10)
                elif "mac" in food and "cheese" in food:
                    if unit == "":
                        factor = 2.5  # Assume one serving (approx. 250g)
                return {
                    "calories": round(std_nutr["calories"] * factor, 2),
                    "protein": round(std_nutr["protein"] * factor, 2),
                    "fat": round(std_nutr["fat"] * factor, 2),
                    "carbs": round(std_nutr["carbs"] * factor, 2)
                }
        
        usda_data = self.usda.search_food(food)
        if usda_data:
            api_calories = float(usda_data.get("calories", 0))
            food_category = self._identify_food_category(food, api_calories)
            reasonable_calorie_ranges = {
                "meat": (100, 300),
                "grain": (100, 200),
                "vegetable": (20, 80),
                "fruit": (40, 100),
                "dairy": (50, 400),
                "oil": (800, 900),
                "beverage": (0, 50),
                "unknown": (50, 300)
            }
            calorie_range = reasonable_calorie_ranges.get(food_category, reasonable_calorie_ranges["unknown"])
            if api_calories < calorie_range[0] * 0.5 or api_calories > calorie_range[1] * 2:
                adjusted_calories = (calorie_range[0] + calorie_range[1]) / 2
                adjustment_ratio = adjusted_calories / max(api_calories, 1)
                return {
                    "calories": round(adjusted_calories * factor, 2),
                    "protein": round(float(usda_data.get("protein", 0)) * adjustment_ratio * factor, 2),
                    "fat": round(float(usda_data.get("fat", 0)) * adjustment_ratio * factor, 2),
                    "carbs": round(float(usda_data.get("carbs", 0)) * adjustment_ratio * factor, 2)
                }
            else:
                max_factor = self._get_reasonable_max_factor(food_category)
                reasonable_factor = min(factor, max_factor)
                return {
                    "calories": round(api_calories * reasonable_factor, 2),
                    "protein": round(float(usda_data.get("protein", 0)) * reasonable_factor, 2),
                    "fat": round(float(usda_data.get("fat", 0)) * reasonable_factor, 2),
                    "carbs": round(float(usda_data.get("carbs", 0)) * reasonable_factor, 2)
                }
        return self._generic_nutrition_estimate(food, factor)

    def _generic_nutrition_estimate(self, food_name: str, factor: float):
        """
        Estimate nutritional values based on the food name when API data is unavailable.
        """
        food = food_name.lower()
        if any(word in food for word in ["chicken", "beef", "pork", "fish", "meat"]):
            cal, pro, fat, carb = 180, 25, 10, 0
        elif any(word in food for word in ["rice", "pasta", "noodle", "bread"]):
            cal, pro, fat, carb = 130, 4, 1, 25
        elif any(word in food for word in ["vegetable", "carrot", "broccoli", "spinach"]):
            cal, pro, fat, carb = 30, 2, 0.3, 6
        elif any(word in food for word in ["fruit", "apple", "orange", "banana"]):
            cal, pro, fat, carb = 60, 0.8, 0.2, 15
        elif any(word in food for word in ["cheese", "milk", "yogurt", "cream", "dairy"]):
            cal, pro, fat, carb = 130, 7, 9, 5
        else:
            cal, pro, fat, carb = 100, 5, 5, 10
            
        reasonable_factor = min(factor, 8)
        return {
            "calories": round(cal * reasonable_factor, 2),
            "protein": round(pro * reasonable_factor, 2),
            "fat": round(fat * reasonable_factor, 2),
            "carbs": round(carb * reasonable_factor, 2)
        }

    def classify_ingredients(self, ing_list):
        """
        Classify ingredients into main ingredients and secondary ingredients.
        If an ingredient contains any of the main keywords, it is considered main;
        if it contains any of the secondary keywords, it is considered secondary;
        default is main.
        """
        main_ings = []
        sub_ings = []
        for ing in ing_list:
            lower_ing = ing.lower()
            if any(mk in lower_ing for mk in self.main_keywords):
                main_ings.append(ing)
            elif any(sk in lower_ing for sk in self.sub_keywords):
                sub_ings.append(ing)
            else:
                main_ings.append(ing)
        return main_ings, sub_ings

    def _analyze_ingredients_list(self, ing_list, dish_name="Standard Recipe"):
        """
        Analyze the given ingredient list to compute detailed nutritional information
        and apply a reasonability check.
        """
        details = []
        total_cal, total_pro, total_fat, total_carb = 0, 0, 0, 0
        
        dish_name_lower = dish_name.lower()
        for key in self.standard_nutrition:
            if (key == dish_name_lower or key in dish_name_lower or dish_name_lower in key):
                std_nutr = self.standard_nutrition[key]
                portion = 2.5  # Assume a portion is 2.5 times 100g
                if "mac" in dish_name_lower and "cheese" in dish_name_lower:
                    portion = 3.0  # Mac and cheese is typically 300g per serving
                elif "curry" in dish_name_lower:
                    portion = 4.0  # Curry is usually about 400g per serving
                elif "coke" in dish_name_lower or "cola" in dish_name_lower:
                    portion = 3.3  # A can of coke is 330ml
                
                simplified_details = [{
                    "food_name": key,
                    "quantity": 1,
                    "unit": "serving",
                    "calories": round(std_nutr["calories"] * portion, 2),
                    "protein": round(std_nutr["protein"] * portion, 2),
                    "fat": round(std_nutr["fat"] * portion, 2),
                    "carbs": round(std_nutr["carbs"] * portion, 2)
                }]
                return {
                    "type": "dish",
                    "dish_name": dish_name,
                    "ingredients": simplified_details,
                    "total_nutrition": {
                        "calories": round(std_nutr["calories"] * portion, 2),
                        "protein": round(std_nutr["protein"] * portion, 2),
                        "fat": round(std_nutr["fat"] * portion, 2),
                        "carbs": round(std_nutr["carbs"] * portion, 2)
                    }
                }
        
        for ing in ing_list:
            qty, unit, fname = self.parse_ingredient(ing)
            nutr = self.get_food_nutrition(fname, qty, unit)
            details.append({
                "food_name": fname,
                "quantity": round(qty, 2),
                "unit": unit,
                "calories": nutr["calories"],
                "protein": nutr["protein"],
                "fat": nutr["fat"],
                "carbs": nutr["carbs"]
            })
            total_cal += nutr["calories"]
            total_pro += nutr["protein"]
            total_fat += nutr["fat"]
            total_carb += nutr["carbs"]
        
        # Set a reasonable upper calorie limit based on the dish type
        reasonable_upper_limit = 1200  # Default upper limit
        if "curry" in dish_name_lower:
            reasonable_upper_limit = 900
        elif "mac" in dish_name_lower and "cheese" in dish_name_lower:
            reasonable_upper_limit = 700
        elif "salad" in dish_name_lower:
            reasonable_upper_limit = 500
        elif "soup" in dish_name_lower:
            reasonable_upper_limit = 400

        if total_cal > reasonable_upper_limit * 2:
            adjustment_factor = reasonable_upper_limit / total_cal
            for item in details:
                item["calories"] *= adjustment_factor
                item["protein"] *= adjustment_factor
                item["fat"] *= adjustment_factor
                item["carbs"] *= adjustment_factor
            total_cal = reasonable_upper_limit
            total_pro *= adjustment_factor
            total_fat *= adjustment_factor
            total_carb *= adjustment_factor
        
        return {
            "type": "dish",
            "dish_name": dish_name,
            "ingredients": details,
            "total_nutrition": {
                "calories": round(total_cal, 2),
                "protein": round(total_pro, 2),
                "fat": round(total_fat, 2),
                "carbs": round(total_carb, 2)
            }
        }

    def _process_multiple(self, text: str):
        """
        When multiple ingredients are provided (non-standard recipe mode),
        directly parse and calculate the nutritional values.
        """
        if self.method == "llm":
            extracted = self.llm_extractor.extract_food_with_llm(text)
        elif self.method == "ner":
            extracted = self.ner_extractor.extract_food_with_ner(text)
        else:
            n = self.ner_extractor.extract_food_with_ner(text)
            l = self.llm_extractor.extract_food_with_llm(text)
            extracted = self._merge_ner_llm(n, l)
        
        # Extra cleanup to ensure each food item is free of unwanted markers
        if self.ner_extractor and hasattr(self.ner_extractor, 'clean_entity_text'):
            for item in extracted:
                item["food"] = self.ner_extractor.clean_entity_text(item["food"])
        
        if not extracted:
            return None
            
        if len(extracted) == 1:
            f = extracted[0]
            try:
                q = float(f["quantity"])
            except:
                q = 1.0
            nutr = self.get_food_nutrition(f["food"], q)
            return {
                "type": "single",
                "items": [{
                    "food_name": f["food"],
                    "quantity": q,
                    "unit": "",
                    "calories": nutr["calories"],
                    "protein": nutr["protein"],
                    "fat": nutr["fat"],
                    "carbs": nutr["carbs"]
                }]
            }
        else:
            details = []
            tc, tp, tf, th = 0, 0, 0, 0
            for f in extracted:
                fd = f["food"]
                try:
                    q = float(f["quantity"])
                except:
                    q = 1.0
                nutr = self.get_food_nutrition(fd, q)
                details.append({
                    "food_name": fd,
                    "quantity": q,
                    "unit": "",
                    "calories": nutr["calories"],
                    "protein": nutr["protein"],
                    "fat": nutr["fat"],
                    "carbs": nutr["carbs"]
                })
                tc += nutr["calories"]
                tp += nutr["protein"]
                tf += nutr["fat"]
                th += nutr["carbs"]
            return {
                "type": "multiple",
                "detailed_breakdown": details,
                "total_nutrition": {
                    "calories": round(tc, 2),
                    "protein": round(tp, 2),
                    "fat": round(tf, 2),
                    "carbs": round(th, 2)
                }
            }

    def _merge_ner_llm(self, ner_foods, llm_foods):
        merged = {}
        for x in ner_foods:
            fd = x["food"].strip().lower()
            try:
                q = float(x["quantity"])
            except:
                q = 1.0
            merged[fd] = max(merged.get(fd, 0), q)
        for x in llm_foods:
            fd = x["food"].strip().lower()
            try:
                q = float(x["quantity"])
            except:
                q = 1.0
            merged[fd] = max(merged.get(fd, 0), q)
        final = []
        for fd, q in merged.items():
            final.append({"food": fd, "quantity": q})
        return final

    def process_text(self, text: str, use_standard_recipe: bool = True):
        """
        Main function: Parse the input text and extract food information.
        """
        dish_names = self.extract_dish_names(text)
        
        if not dish_names or not use_standard_recipe:
            return self._process_multiple(text)
        
        dishes_found = []
        other_foods = []
        
        for dish_name in dish_names:
            if self.is_probably_dish(dish_name):
                db_title, db_ings = self.search_recipe_in_db(dish_name)
                if db_ings:
                    main_ings, sub_ings = self.classify_ingredients(db_ings)
                    merged = main_ings + sub_ings
                    dishes_found.append({
                        "type": "dish",
                        "dish_name": db_title if db_title else dish_name,
                        "ingredients": merged,
                        "source": "database"
                    })
                else:
                    llm_ings = self.generate_recipe_with_llm(dish_name)
                    if llm_ings:
                        main_ings, sub_ings = self.classify_ingredients(llm_ings)
                        merged = main_ings + sub_ings
                        dishes_found.append({
                            "type": "dish",
                            "dish_name": dish_name,
                            "ingredients": merged,
                            "source": "llm"
                        })
                    else:
                        other_foods.append({"food": dish_name, "quantity": 1.0})
            else:
                other_foods.append({"food": dish_name, "quantity": 1.0})
        
        if self.method != "llm" and self.ner_extractor is not None:
            extracted_foods = self.ner_extractor.extract_food_with_ner(text)
            for food in extracted_foods:
                food_name = food["food"].lower()
                if not any(food_name in dish["dish_name"] for dish in dishes_found):
                    is_duplicate = False
                    for existing_food in other_foods:
                        if food_name == existing_food["food"]:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        other_foods.append(food)
        
        if not dishes_found and not other_foods:
            return self._process_multiple(text)
        
        if len(dishes_found) == 1 and not other_foods:
            return dishes_found[0]
            
        return {
            "type": "combined",
            "dishes": dishes_found,
            "other_foods": other_foods
        }
