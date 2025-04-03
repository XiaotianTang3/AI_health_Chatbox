import ollama
import json
import re

class FoodExtractor:
    def __init__(self, model="phi4"):
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

        Return in the format:
        [               
            {{"food": "food_name", "quantity": quantity}},
            {{"food": "food_name", "quantity": quantity}}
        ]
"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        extracted_text = response["message"]["content"]

        # Try to extract JSON from LLM output
        try:
            food_data = json.loads(re.search(r"\[.*\]", extracted_text, re.DOTALL).group(0))
        except (json.JSONDecodeError, AttributeError):
            food_data = []

        return food_data

# Example usage
if __name__ == "__main__":
    extractor = FoodExtractor()

    user_input = """
    I am so happy today. I went out and met two friends. Then I bought a pair of shoes.
    For breakfast, I had two eggs, a glass of orange juice, and some toast.
    Later, I had a bowl of rice with chicken for lunch.
    """
    
    extracted_foods = extractor.extract_food_with_llm(user_input)
    print(f"Extracted foods: {extracted_foods}")
