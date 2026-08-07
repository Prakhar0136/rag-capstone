import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document

load_dotenv()

def parse_and_clean_pdf(file_path):
    print(f"Parsing {file_path} with Unstructured...")
    
    # 1. Parse the PDF. 
    # The "fast" strategy forces it to read the embedded text rather than using slow OCR image scanning.
    elements = partition_pdf(filename=file_path, strategy="fast")
    
    # 2. Filter out the boilerplate noise
    cleaned_elements = []
    categories_to_ignore = ["Header", "Footer", "PageNumber", "PageBreak"]
    
    for element in elements:
        # Every element detected by Unstructured has a 'category' tag
        if element.category not in categories_to_ignore:
            cleaned_elements.append(element)
            
    print(f"Original elements found: {len(elements)}")
    print(f"Cleaned elements kept: {len(cleaned_elements)}")

    # 3. Save the clean text to a file so we can visually verify it
    output_file = "cleaned_data.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for element in cleaned_elements:
            f.write(str(element) + "\n\n")
            
    print(f"\nSuccess! Open {output_file} in VS Code to see your clean data.")
    
    # 4. Convert our cleaned text back into a single LlamaIndex Document for tomorrow's work
    full_text = "\n".join([str(el) for el in cleaned_elements])
    return [Document(text=full_text)]

if __name__ == "__main__":
    # CHANGE THIS to the actual name of your PDF!
    clean_documents = parse_and_clean_pdf("./data/sample.pdf")