from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Initialize the embedding model
# This will download the model to your machine on the first run
model_name = "BAAI/bge-small-en-v1.5"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name)

# 2. Mocking your chunks from Phase 2
# (Replace this with the actual list of strings from your chunker)
document_chunks = [
    "Machine learning is a field of study in artificial intelligence.",
    "A neural network is designed to mimic the human brain.",
    "The stock market experienced a significant dip on Tuesday."
]

# 3. Generate the embeddings
print("Generating embeddings...")
# model.encode automatically handles tokenization and batching
embeddings = model.encode(document_chunks, show_progress_bar=True)

# 4. Verify the output
print("\n--- Verification ---")
print(f"Number of chunks: {len(document_chunks)}")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])} (Expected 384 for bge-small)")
print(f"Data type: {type(embeddings)} (Expected numpy.ndarray)")

# Optional: Look at a tiny slice of the first vector
print(f"\nFirst 5 values of Chunk 1's vector: \n{np.round(embeddings[0][:5], 4)}")