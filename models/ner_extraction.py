import spacy
from typing import List, Dict
import re

class NERFoodExtractor:
    def __init__(self, model_path="food_ner_model"):
        """
        Load a trained spaCy NER model.
        Assumes the trained model is stored in the 'food_ner_model' folder.
        """
        try:
            self.nlp = spacy.load(model_path)  # Load custom NER model
            print(f"Loaded custom food NER model from {model_path}")
        except:
            print("Warning: Custom NER model not found, falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Time markers and other non-food words to filter out
        self.time_markers = ["today", "yesterday", "tomorrow", "morning", "evening", "night", 
                             "afternoon", "last night", "this morning", "now", "later"]
    
    def clean_entity_text(self, text: str) -> str:
        """
        Clean entity text by removing time markers and other non-food words.
        """
        text = text.lower().strip()
        
        # Remove trailing time markers
        for marker in self.time_markers:
            if text.endswith(" " + marker):
                text = text.replace(" " + marker, "")
            # Also remove leading time markers
            if text.startswith(marker + " "):
                text = text.replace(marker + " ", "")
                
        return text
        
    def extract_food_with_ner(self, text: str) -> List[Dict[str, float]]:
        """
        Extract food names and quantities using a spaCy model. Returns a list such as:
        [
          {"food": "eggs", "quantity": 2},
          {"food": "milk", "quantity": 1}
        ]
        """
        doc = self.nlp(text)
        results = []
        current_quantity = None
        food_entities = []
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ == "QUANTITY":
                # e.g., "2 cups"
                quantity_str = ent.text
                current_quantity = self._parse_quantity_str(quantity_str)
            elif ent.label_ == "FOOD":
                # Clean entity text by removing time markers
                clean_food = self.clean_entity_text(ent.text)
                
                # Skip if cleaned food name is empty or too short
                if len(clean_food) < 2:
                    continue
                    
                # Pair with previously found quantity or default to 1
                qty = current_quantity if current_quantity is not None else 1
                food_entities.append({"food": clean_food, "quantity": qty})
                current_quantity = None
        
        # If no food entities found, attempt regex matching
        if not food_entities:
            food_patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:cups?|tbsps?|tsps?|ounces?|oz|pounds?|lbs?|grams?|g|kilograms?|kg)?\s+of\s+([a-zA-Z\s]+)',
                r'([a-zA-Z\s]+)\s+(\d+(?:\.\d+)?)\s*(?:cups?|tbsps?|tsps?|ounces?|oz|pounds?|lbs?|grams?|g|kilograms?|kg)?'
            ]
            
            for pattern in food_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        if match.group(1).isdigit() or self._is_float(match.group(1)):
                            qty = float(match.group(1))
                            food = match.group(2).strip().lower()
                        else:
                            food = match.group(1).strip().lower()
                            qty = float(match.group(2)) if self._is_float(match.group(2)) else 1.0
                        
                        # Clean extracted food name
                        food = self.clean_entity_text(food)
                        if len(food) >= 2:  # Ensure meaningful food name
                            food_entities.append({"food": food, "quantity": qty})
        
        # If still no matches, try analyzing words for common foods
        if not food_entities:
            common_foods = ["apple", "banana", "chicken", "rice", "pasta", "eggs", "milk", 
                            "beef", "pork", "fish", "bread", "cheese", "yogurt", "salad",
                            "pizza", "burger", "sandwich", "soup", "stew", "curry", "noodles", "steak"]
            
            # Split text into words and check
            words = text.lower().split()
            for food in common_foods:
                if food in words:
                    food_entities.append({"food": food, "quantity": 1.0})
        
        # Remove duplicates and filter invalid entities
        food_dict = {}
        for entity in food_entities:
            food = entity["food"]
            
            # Skip if it's just a time marker
            if food in self.time_markers:
                continue
                
            # Add to dictionary, keeping the highest quantity
            if food in food_dict:
                food_dict[food] = max(food_dict[food], entity["quantity"])
            else:
                food_dict[food] = entity["quantity"]
        
        # Convert back to list format
        for food, qty in food_dict.items():
            results.append({"food": food, "quantity": qty})
        
        return results
    
    def _parse_quantity_str(self, quantity_str: str) -> float:
        """
        Extract numeric value from a string, e.g., "2 cups" -> 2.0.
        Returns 1.0 if no numeric value is found.
        """
        match = re.search(r"\d+(\.\d+)?", quantity_str)
        if match:
            return float(match.group(0))
        
        # Handle fractions
        if "/" in quantity_str:
            try:
                parts = quantity_str.split("/")
                return float(parts[0]) / float(parts[1])
            except:
                pass
        return 1.0
    
    def _is_float(self, s):
        try:
            float(s)
            return True
        except:
            return False
