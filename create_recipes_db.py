import sqlite3
import json

# Connect to SQLite database (or create it)
conn = sqlite3.connect('recipes.db')
cursor = conn.cursor()

# Create the recipes table with columns for id, title, ingredients, and instructions
cursor.execute('''
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY,
    title TEXT,
    ingredients TEXT,  -- Store ingredients as a JSON string
    instructions TEXT  -- Store instructions as a JSON string
)
''')
conn.commit()

# Load data into the table from layer1.json
with open('data.json', 'r', encoding='utf-8') as f:
    recipes_data = json.load(f)

# Insert each recipe into the database
for recipe in recipes_data:
    recipe_id = recipe.get("id", "")
    title = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    instructions = recipe.get("instructions", [])

    # Serialize ingredients and instructions into JSON strings
    ingredients_json = json.dumps(ingredients, ensure_ascii=False)
    instructions_json = json.dumps(instructions, ensure_ascii=False)

    try:
        cursor.execute('''
        INSERT INTO recipes (id, title, ingredients, instructions)
        VALUES (?, ?, ?, ?)
        ''', (id, title, ingredients_json, instructions_json))
    except Exception as e:
        print(f"Error inserting recipe {id}: {e}")

conn.commit()
conn.close()
print("Database created and recipes inserted successfully!")