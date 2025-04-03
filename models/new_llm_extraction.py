# models/llm_extraction.py
import ollama
import json
import re

class FoodExtractor:
    def __init__(self, model="mistral"):
        """ Initialize LLM-based food extractor """
        self.model = model

    def extract_food_with_llm(self, text):
        """
        Use LLM to extract food items and quantities from text.
        """
        prompt = f"""
        Extract food items and their respective quantities from this text:
        "{text}"

        Rules:
        1. Only extract food items (ignore modifiers like "with", "on", "along with").
        2. If the quantity is unclear, assume it is 1.
        3. Output a clean JSON format with only "food" and "quantity" fields.
        4. Convert all fractions to decimals (e.g., 1/2 to 0.5).
        5. Identify dish names (like "pizza", "salad", "mac and cheese") as single food items.

        Return in the format:
        [               
            {{"food": "food_name", "quantity": quantity}},
            {{"food": "food_name", "quantity": quantity}}
        ]
        """

        try:
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            extracted_text = response["message"]["content"]

            # Try to extract JSON from LLM output
            matches = re.search(r"\[.*\]", extracted_text, re.DOTALL)
            if matches:
                food_data = json.loads(matches.group(0))
                return food_data
            else:
                # Try a simpler approach - look for anything that could be JSON
                for line in extracted_text.split('\n'):
                    if line.strip().startswith('[') and line.strip().endswith(']'):
                        try:
                            return json.loads(line.strip())
                        except:
                            pass
                return []
                
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []