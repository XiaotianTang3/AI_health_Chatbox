#%% [markdown]
# ## Data Preprocessing Plan
# 1. Load and Inspect Data
# 2. Clean Each Dataset (handle missing values, fix inconsistencies)
# 3. Merge Key Datasets:
# - `food.csv` (Food items)
# - `nutrient.csv`(Nutrient names and units)
# `food_nutrient.csv`(Links food items to nutrients)
# `food_portion.csv` (For portion size conversion)
# 4. Save the Cleaned Data for Use

#%%
# 1. Load and Inspect Data
import pandas as pd

# Load the datasets
food = pd.read_csv("food.csv")
nutrient = pd.read_csv("nutrient.csv")
food_nutrient = pd.read_csv("food_nutrient.csv")
food_portion = pd.read_csv("food_portion.csv")

# Display basic info
print("Food Data Info:")
print(food.info(), "\n")

print("Nutrient Data Info:")
print(nutrient.info(), "\n")

print("Food Nutrient Data Info:")
print(food_nutrient.info(), "\n")

print("Food Portion Data Info:")
print(food_portion.info(), "\n")

# Check for missing values
print("Missing Values in Food Dataset:\n", food.isnull().sum(), "\n")
print("Missing Values in Nutrient Dataset:\n", nutrient.isnull().sum(), "\n")
print("Missing Values in Food Nutrient Dataset:\n", food_nutrient.isnull().sum(), "\n")
print("Missing Values in Food Portion Dataset:\n", food_portion.isnull().sum(), "\n")

# %%
# 2. Clean Each Dataset
# (A) Food Dataset
# Remove duplicates
food.drop_duplicates(inplace=True)

# Drop rows with missing food descriptions
food.dropna(subset=['description'], inplace=True)

# Convert column names to lowercase for consistency
food.columns = food.columns.str.lower().str.replace(" ", "_")

print("Food data cleaned!")

# (B) Nutrient Dataset
# Remove duplicates
nutrient.drop_duplicates(inplace=True)

# Drop irrelevant columns if needed (e.g., "rank" if not useful)
nutrient.drop(columns=['rank'], inplace=True, errors='ignore')

# Standardize column names
nutrient.columns = nutrient.columns.str.lower().str.replace(" ", "_")

print("Nutrient data cleaned!")

# (C) Food Nutrient Dataset
# Remove duplicates
food_nutrient.drop_duplicates(inplace=True)

# Drop irrelevant columns
food_nutrient.drop(columns=['data_points', 'derivation_id', 'min', 'max', 'median', 'footnote', 'min_year_acquired'], inplace=True, errors='ignore')

# Standardize column names
food_nutrient.columns = food_nutrient.columns.str.lower().str.replace(" ", "_")

print("Food Nutrient data cleaned!")

# (D) Food Portion Dataset
# Remove duplicates
food_portion.drop_duplicates(inplace=True)

# Drop irrelevant columns
food_portion.drop(columns=['footnote', 'min_year_acquired'], inplace=True, errors='ignore')

# Standardize column names
food_portion.columns = food_portion.columns.str.lower().str.replace(" ", "_")

print("Food Portion data cleaned!")


# %%
# 3. Merge Key Datasets
# Merge food_nutrient with nutrient names
food_nutrient_merged = food_nutrient.merge(nutrient, left_on="nutrient_id", right_on="id", how="left")

# Merge food_nutrient with food descriptions
food_nutrient_merged = food_nutrient_merged.merge(food, left_on="fdc_id", right_on="fdc_id", how="left")

# Select only relevant columns
food_nutrient_merged = food_nutrient_merged[['fdc_id', 'description', 'name', 'amount', 'unit_name']]

# Rename columns for clarity
food_nutrient_merged.rename(columns={'name': 'nutrient_name', 'unit_name': 'unit'}, inplace=True)

# Keep only the common nutrients
common_nutrients = ['Protein', 'Total lipid (fat)', 'Carbohydrate, by difference', 'Energy', 'Sugars, total including NLEA', 'Fiber, total dietary', 'Calcium, Ca', 'Iron, Fe', 'Sodium, Na', 'Vitamin C, total ascorbic acid', 'Vitamin A, IU', 'Fatty acids, total saturated', 'Fatty acids, total monounsaturated', 'Fatty acids, total polyunsaturated', 'Cholesterol']
food_nutrient_merged = food_nutrient_merged[food_nutrient_merged['nutrient_name'].isin(common_nutrients)]

# Search for repetitive name of each food and only keep the ones with smallest fdc_id
# such as 'Milk, reduced fat, fluid, 2% milkfat, with added vitamin A and vitamin D' and '2% Milk'
food_nutrient_merged = food_nutrient_merged.sort_values(by=['fdc_id'])
food_nutrient_merged = food_nutrient_merged.drop_duplicates(subset=['description', 'nutrient_name'], keep='first')

print("Merged dataset preview:")
print(food_nutrient_merged.head())

# %%
# Save the cleaned and merged data
food_nutrient_merged.to_csv("cleaned_food_nutrients.csv", index=False)

print("Merged data saved successfully!")

# %%
import os
# Load datasets
interactions_train = pd.read_csv(f"..{os.sep}Food_Recipe_DATA{os.sep}interactions_train.csv")
interactions_test = pd.read_csv(f"..{os.sep}Food_Recipe_DATA{os.sep}interactions_test.csv")
interactions_validation = pd.read_csv(f"..{os.sep}Food_Recipe_DATA{os.sep}interactions_validation.csv")
raw_interactions = pd.read_csv(f"..{os.sep}Food_Recipe_DATA{os.sep}RAW_interactions.csv")
raw_recipes = pd.read_csv(f"..{os.sep}Food_Recipe_DATA{os.sep}RAW_recipes.csv")
pp_recipes = pd.read_csv(f"..{os.sep}Food_Recipe_DATA{os.sep}PP_recipes.csv")

# Step 1: Handle missing values
datasets = {
    "interactions_train": interactions_train,
    "interactions_test": interactions_test,
    "interactions_validation": interactions_validation,
    "raw_interactions": raw_interactions,
    "raw_recipes": raw_recipes,
    "pp_recipes": pp_recipes,
}

for name, df in datasets.items():
    df.fillna("", inplace=True)

#%%
# Step 2: Normalize text data
raw_recipes["name"] = raw_recipes["name"].str.lower().str.strip()
raw_recipes["ingredients"] = raw_recipes["ingredients"].astype(str).str.lower().str.strip()
raw_recipes["steps"] = raw_recipes["steps"].astype(str).str.lower().str.strip()

pp_recipes["name_tokens"] = pp_recipes["name_tokens"].astype(str).str.lower().str.strip()
pp_recipes["ingredient_tokens"] = pp_recipes["ingredient_tokens"].astype(str).str.lower().str.strip()
pp_recipes["steps_tokens"] = pp_recipes["steps_tokens"].astype(str).str.lower().str.strip()

# Step 3: Remove duplicates
for name, df in datasets.items():
    df.drop_duplicates(inplace=True)

#%%
# Step 4: Merge recipe datasets
merged_recipes = raw_recipes.merge(pp_recipes, left_on="id", right_on="id", how="left")

# Step 5: Merge interactions with recipes
interactions_train = interactions_train.merge(merged_recipes, left_on="recipe_id", right_on="id", how="inner")
interactions_test = interactions_test.merge(merged_recipes, left_on="recipe_id", right_on="id", how="inner")
interactions_validation = interactions_validation.merge(merged_recipes, left_on="recipe_id", right_on="id", how="inner")
raw_interactions = raw_interactions.merge(merged_recipes, left_on="recipe_id", right_on="id", how="inner")

#%%
# Step 6: Select relevant columns
merged_recipes = merged_recipes[["id", "name", "ingredients", "steps", "name_tokens", "ingredient_tokens", "steps_tokens"]]
interactions_train = interactions_train[["user_id", "recipe_id", "rating", "name", "ingredients", "steps"]]
interactions_test = interactions_test[["user_id", "recipe_id", "rating", "name", "ingredients", "steps"]]
interactions_validation = interactions_validation[["user_id", "recipe_id", "rating", "name", "ingredients", "steps"]]
raw_interactions = raw_interactions[["user_id", "recipe_id", "rating", "name", "ingredients", "steps"]]

#%%
# Step 7: Display cleaned merged data samples
print("Merged Recipe Data Sample:")
print(merged_recipes.head())

print("\nMerged Training Interactions Sample:")
print(interactions_train.head())

print("\nMerged Test Interactions Sample:")
print(interactions_test.head())

print("\nMerged Validation Interactions Sample:")
print(interactions_validation.head())

print("\nMerged Raw Interactions Sample:")
print(raw_interactions.head())


# %%
# Step 8: Save cleaned data
merged_recipes.to_csv("cleaned_merged_recipes.csv", index=False)
interactions_train.to_csv("cleaned_interactions_train.csv", index=False)
interactions_test.to_csv("cleaned_interactions_test.csv", index=False)
interactions_validation.to_csv("cleaned_interactions_validation.csv", index=False)
raw_interactions.to_csv("cleaned_raw_interactions.csv", index=False)

# %%
