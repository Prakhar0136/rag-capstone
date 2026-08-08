import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
# NEW: Import LlamaIndex HuggingFace Embedding wrapper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Initialize the embedding model (downloads locally)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def parse_and_clean_pdf(file_path):
    print(f"Parsing {file_path} with Unstructured...")
    elements = partition_pdf(filename=file_path, strategy="fast")
    
    cleaned_elements = []
    categories_to_ignore = ["Header", "Footer", "PageNumber", "PageBreak"]
    
    for element in elements:
        if element.category not in categories_to_ignore:
            cleaned_elements.append(element)
            
    full_text = "\n".join([str(el) for el in cleaned_elements])
    return [Document(text=full_text)]

def chunk_documents(documents):
    print("Chunking documents into smaller pieces...")
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Successfully split into {len(nodes)} chunks.")
    return nodes

def generate_embeddings_for_chunks(nodes):
    print("Generating embeddings for PDF chunks...")
    for node in nodes:
        # Generate vector for each chunk's text and attach it to the node
        node.embedding = embed_model.get_text_embedding(node.get_content())
    
    print(f"Done! Sample vector length: {len(nodes[0].embedding)}")
    return nodes

if __name__ == "__main__":
    # 1. Parse and clean
    clean_docs = parse_and_clean_pdf("./data/sample.pdf")
    
    # 2. Chunk
    chunks = chunk_documents(clean_docs)
    
    # 3. Generate Embeddings
    embedded_chunks = generate_embeddings_for_chunks(chunks)