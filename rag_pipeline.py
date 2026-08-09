import os
from dotenv import load_dotenv
import qdrant_client

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# 1. Load your hidden API keys
load_dotenv()

def run_rag_pipeline(user_query):
    print("1. Initializing Embedding Model and LLM...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # We use llama3-8b-8192 for high-speed, accurate generation
    llm = Groq(model="openai/gpt-oss-20b")

    print("2. Connecting to Qdrant Database...")
    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(client=client, collection_name="capstone_docs")

    # Connect the LlamaIndex view to the existing database
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    print("3. Building the Strict System Prompt...")
    # This is the most critical part of preventing AI hallucinations
    qa_prompt_tmpl_str = (
        "You are a helpful assistant. Answer the user's question USING ONLY the provided context.\n"
        "If the context does not contain the answer, say 'I do not know.'\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    print(f"\n4. Executing Pipeline for Query: '{user_query}'\n")
    # Assemble the engine: it handles the embedding, the retrieval, the prompt injection, and the LLM call!
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        text_qa_template=qa_prompt_tmpl
    )

    # 5. Get the final synthesized answer
    response = query_engine.query(user_query)
    
    print("=" * 60)
    print("FINAL AI RESPONSE:")
    print("=" * 60)
    print(response.response)
    print("=" * 60)

    # (Optional) See what exact chunks the AI used to write its answer
    print("\n[Sources Used]")
    for source_node in response.source_nodes:
        print(f"- Source Match Score: {source_node.score:.4f}")

if __name__ == "__main__":
    # Test a question that should definitely be in your PDF
   
    
    
    run_rag_pipeline("Who won the Super Bowl in 2022?")