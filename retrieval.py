import os
from dotenv import load_dotenv
import qdrant_client
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

def search_knowledge_base(query_text, top_k=3):
    print(f"1. Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("2. Connecting to local Qdrant collection...")
    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(client=client, collection_name="capstone_docs")

    # Reconstruct index view from existing Qdrant collection
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    print(f"3. Searching for top {top_k} relevant chunks for query: '{query_text}'...\n")
    retriever = index.as_retriever(similarity_top_k=top_k)
    
    # Perform vector search
    results = retriever.retrieve(query_text)

    # Display results
    print("=" * 60)
    print(f"RETRIEVAL RESULTS FOR: '{query_text}'")
    print("=" * 60)

    for idx, node_with_score in enumerate(results, start=1):
        score = node_with_score.score
        content = node_with_score.node.get_content()
        
        print(f"\n Rank {idx} | Similarity Score: {score:.4f}")
        print("-" * 60)
        print(content.strip())
        print("-" * 60)

if __name__ == "__main__":
    # CHANGE THIS to a specific question answered inside your PDF!
    sample_query = "What is the maximum acceleration range of the PhidgetAccelerometer?"
    search_knowledge_base(sample_query, top_k=3)