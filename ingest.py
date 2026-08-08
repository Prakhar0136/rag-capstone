import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

load_dotenv()

# 1. Load local embedding model (BAAI/bge-small-en-v1.5 produces 384-dim vectors)
print("Initializing HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def parse_and_clean_pdf(file_path):
    print(f"\n1. Parsing '{file_path}' with Unstructured...")
    elements = partition_pdf(filename=file_path, strategy="fast")
    
    cleaned_elements = []
    categories_to_ignore = ["Header", "Footer", "PageNumber", "PageBreak"]
    
    for element in elements:
        if element.category not in categories_to_ignore:
            cleaned_elements.append(element)
            
    full_text = "\n".join([str(el) for el in cleaned_elements])
    return [Document(text=full_text)]

def ingest_to_qdrant(documents):
    print("\n2. Chunking document into 500-token pieces...")
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks.")

    print("\n3. Connecting to local Qdrant instance...")
    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    
    # 4. Connect LlamaIndex to Qdrant vector collection
    vector_store = QdrantVectorStore(client=client, collection_name="capstone_docs")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("\n4. Embedding chunks and indexing into Qdrant...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("\n Success! All document chunks are stored in Qdrant.")
    return index

if __name__ == "__main__":
    # UPDATE THIS to match your exact file name inside the data/ folder
    pdf_path = "./data/sample.pdf" 
    
    docs = parse_and_clean_pdf(pdf_path)
    ingest_to_qdrant(docs)