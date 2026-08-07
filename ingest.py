import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

# 1. Load your secret keys (good practice to include in every main script)
load_dotenv()

def load_data():
    print("Scanning the 'data' folder for documents...")
    
    # 2. Initialize the reader pointing to your data folder
    reader = SimpleDirectoryReader(input_dir="./data")
    
    # 3. Extract the text and create Document objects
    documents = reader.load_data()
    
    print(f"Successfully loaded {len(documents)} pages/items.")
    
    # 4. Prove it worked by printing the first 500 characters of the first page
    if documents:
        print("\n--- Text Preview (First Page) ---")
        print(documents[0].text[:500] + "...\n")
        
        print("--- Hidden Metadata ---")
        print(documents[0].metadata)

if __name__ == "__main__":
    load_data()