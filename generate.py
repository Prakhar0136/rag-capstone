import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# 1. Load the hidden API keys from .env
load_dotenv()

def test_llm():
    print("Connecting to Groq...")
    
    # 2. Initialize the Llama 3 model
    # LlamaIndex automatically looks for GROQ_API_KEY in your environment variables
    llm = Groq(model="openai/gpt-oss-20b")
    
    # 3. Create a hardcoded test prompt
    prompt = "Explain Retrieval-Augmented Generation (RAG) in one simple sentence."
    print(f"\nSending Prompt: '{prompt}'")
    
    # 4. Generate the response
    print("Waiting for response...")
    response = llm.complete(prompt)
    
    # 5. Print the output
    print("\n" + "="*50)
    print("AI RESPONSE:")
    print("="*50)
    print(response.text)
    print("="*50)

if __name__ == "__main__":
    test_llm()