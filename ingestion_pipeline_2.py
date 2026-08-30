import os
import json
from typing import List

# Langchain components
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from dotenv import load_dotenv

load_dotenv()

# partition document into atomic elements
def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path, # Path to PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"], # Grab images found in PDF
        extract_image_block_to_payload=True # Store images as base64 data we can actually use
    )

    print(f"Extracted {len(elements)} elements")
    return elements

# create thunk by title strategy
def create_chunks_by_title(elements):
     """Create intelligent chunks using title-based strategy"""
     print("Creating smart chunks...")

     chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 characters with neighbors
     )

     print(f"Created {len(chunks)} chunks")

     return chunks

# Separate content types
def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    # Check for tables and images in original elements
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == "Table":
                content_data['types'].append('table')
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            # Handle images
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data

# Create AI enhanced summary for the tables/images content
def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""

    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Build the text prompt
        prompt_text = f"""You are Creating a searchable description for document content retrieval.
        
        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}
        """

        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"

            for i, table in enumerate(tables):
                prompt_text += f"TABLE {i+1}:\n{table}\n\n"

                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, pattern in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findibility over brevity.

                SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the messages
        for image_base64 in images:
            message_content.append({
                "type": "image_url", 
                "image_url": {
                "url": f"data:image/png;base64,{image_base64}",
            }})

        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"Error creating AI-enhanced summary: {e}")
        return text[:300]  # Fallback to original text if AI fails

# Summarise chunks using AI
def summarise_chunks(chunks):
    """Process all chunks with AI Summaries..."""
    print("Processing chunks with AI Summaries...")

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"Processing chunk {current_chunk}/{total_chunks}")

        # Analyze chunk content
        content_data = separate_content_types(chunk)

        # Debug prints
        print(f"Types found: {content_data['types']}")
        print(f"Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        # Create AI-enhanced summary if chunk has tables/images
        if content_data["tables"] or content_data["images"]:
            print(f"Creating AI summary for mixed content...")
            
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"],
                    content_data["tables"],
                    content_data["images"],
                )

                print(f"AI summary created successfully")
                print(f"Enhanced content preview: {enhanced_content[:200]}...")

            except Exception as e:
                print(f"AI summary failed: {e}")
                enhanced_content = content_data["text"]

        else:
            print(f"Using raw text (no tables/images)")
            enhanced_content = content_data["text"]

        # Createa Langchain Document with enhanced content
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables"],
                    "images_base64": content_data["images"],
                })
            }
        )

        langchain_documents.append(doc)

    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents

# Create Vector Store using Chroma
def create_vector_store(documents: List[Document], persist_directory: str = "db/chroma_db"):
    """Create a Chroma vector store from documents"""
    print(f"Creating vector store in directory: {persist_directory}")

    # Initialize embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create Chroma vector store
    print(f"Creating Chroma vector store with {len(documents)} documents...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Finished creating vector store")

    print(f"Vector store created and persisted at: {persist_directory}")

    return vector_store

# Create atomic elements
file_path = "./docs/attention-is-all-you-need.pdf"
elements = partition_document(file_path)

# Create chunks using atomic elements
chunks = create_chunks_by_title(elements)
processed_chunks = summarise_chunks(chunks)

# Create vector store from processed chunks
db = create_vector_store(processed_chunks)

# retrieval
query = "Give me model architecture image and explain the attention mechanism in detail"
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)
print(f"Retrieved {len(chunks)} chunks for query: '{query}'")

def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

        CONTENT TO ANALYZE:
        """
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
        Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

        ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

# Usage
final_answer = generate_final_answer(chunks, query)
print(final_answer)

"""
Basic Architecture Questions

1. What are the two main components of the Transformer architecture? 
2. How many layers does the base Transformer model use in both encoder and decoder? 
3. What is the dimensionality (dmodel) used in the base Transformer model? 

Attention Mechanism Questions

4. What is the formula for Scaled Dot-Product Attention? 
5. Why do the authors scale the dot products by 1/√dk in their attention mechanism? 
6. How many attention heads does the Transformer use, and what is the dimension of each head? 

Comparative Analysis Question

7. According to Table 1, what are the main advantages of self-attention layers compared to recurrent and convolutional layers in terms of computational complexity and parallelization? 
"""