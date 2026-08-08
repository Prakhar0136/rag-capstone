import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document
# NEW: Import the tool that intelligently splits text
from llama_index.core.node_parser import SentenceSplitter 

load_dotenv()

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
    
    # 1. Configure the splitter
    # chunk_size: Target number of tokens per chunk (500 is standard for RAG)
    # chunk_overlap: How many tokens overlap between chunks to preserve context
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    
    # 2. Extract "Nodes" (LlamaIndex's official term for chunks)
    nodes = splitter.get_nodes_from_documents(documents)
    
    print(f"Successfully split the document into {len(nodes)} chunks.")
    
    # 3. Print the first 3 chunks to physically see the overlap and size
    print("\n--- Chunk Preview ---")
    for i, node in enumerate(nodes[:3]):
        print(f"\n[Chunk {i + 1}]")
        print(node.text)
        print("-" * 50)
        
    return nodes

if __name__ == "__main__":
    # 1. Parse and clean the data
    clean_docs = parse_and_clean_pdf("./data/sample.pdf")
    
    # 2. Chunk the cleaned data
    chunks = chunk_documents(clean_docs)