import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FAQRetriever:
    def __init__(self, faq_files, model_name="sentence-transformers/all-MiniLM-L6-v2", embedding_dir="cache/"):
        """
        Initialize the FAQ Retriever:
        - Load FAQ data from multiple CSV files
        - Compute or load precomputed embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_file = os.path.join(embedding_dir, "faq_embeddings.pkl")
        self.faq_data = self.load_faq_data(faq_files)

        # Ensure the cache directory exists
        os.makedirs(embedding_dir, exist_ok=True)

        # Load or compute FAQ embeddings
        if os.path.exists(self.embedding_file):
            self.faq_embeddings = self.load_embeddings()
        else:
            self.build_faq_embeddings()

    def load_faq_data(self, faq_files):
        """ Load multiple FAQ files and merge them into a single DataFrame """
        faq_list = []
        for file in faq_files:
            df = pd.read_csv(file)

            if "Answer" in df.columns:  # KidsHealth series
                df = df[["Question", "Answer"]]
            elif "Article" in df.columns:  # Nutrition FAQ
                df = df[["Question", "Article"]]
                df.rename(columns={"Article": "Answer"}, inplace=True)  # Standardize column name

            faq_list.append(df)

        all_faq = pd.concat(faq_list, ignore_index=True)
        return all_faq

    def build_faq_embeddings(self):
        """ Compute embeddings for all FAQ questions and save them to a file """
        questions = self.faq_data["Question"].tolist()
        self.faq_embeddings = self.model.encode(questions, convert_to_numpy=True)

        # Save embeddings to the 'cache' directory
        with open(self.embedding_file, "wb") as f:
            pickle.dump(self.faq_embeddings, f)
        print(f"FAQ embeddings saved to {self.embedding_file}")

    def load_embeddings(self):
        """ Load precomputed embeddings from the 'cache' directory """
        with open(self.embedding_file, "rb") as f:
            return pickle.load(f)

    def search_faq(self, query, top_k=1):
        """ Search FAQs using cosine similarity and return the most relevant question and answer """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarity_scores = cosine_similarity(query_embedding, self.faq_embeddings)
        best_match_idx = similarity_scores.argmax()
        best_match_question = self.faq_data.iloc[best_match_idx]["Question"]
        best_match_answer = self.faq_data.iloc[best_match_idx]["Answer"]

        response = f"The most relevant question I found is: \"{best_match_question}\"\n"
        response += f"\nAnswer: {best_match_answer}\n"
        response += "\nThis may not be exactly what you were looking for, but it's the closest match I found."
        
        return response

# Example usage
if __name__ == "__main__":
    faq_files = [
        "data/kidshealth_for_kids_faq.csv",
        "data/kidshealth_for_parents_faq.csv",
        "data/kidshealth_for_teens_faq.csv",
        "data/nutrition_faq.csv"
    ]
    
    retriever = FAQRetriever(faq_files)

    query = "What should I eat to stay healthy?"
    answer = retriever.search_faq(query)
    print(f"Chatbot: {answer}")
