import os
import time
from typing import List, Tuple
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken

# Set up your OpenAI API key
openai.api_key = "sk-proj-EG6oHivYBFQ8PNI7NStxT3BlbkFJHi9YGda9NhjdHxwxFZxq"

# Set the model to use for embeddings
embedding_model = "text-embedding-ada-002"

# Function to get the embedding for a piece of text
def get_doc_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    return get_embedding(text, engine=embedding_model)

# Function to compute vector similarity between two documents
def vector_similarity(x: List[float], y: List[float]) -> float:
    """
    We could use cosine_similarity or dot product
    """
    return cosine_similarity(x, y)

# Function to split text into chunks of specified size
def get_text_chunks(text: str, chunk_size: int) -> List[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    end = chunk_size
    while start < len(tokens):
        chunk_text = tokenizer.decode(tokens[start:end])
        chunks.append(chunk_text)
        start = end
        end += chunk_size
    return chunks

# Function to create vector database from text file
def create_vector_db(file_path: str, chunk_size: int = 1000) -> List[Tuple[str, List[float]]]:
    with open(file_path, "r") as f:
        text = f.read()

    chunks = get_text_chunks(text, chunk_size)
    vector_db = []

    for chunk in chunks:
        try:
            embedding = get_doc_embedding(chunk)
            vector_db.append((chunk, embedding))
        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting for 10 seconds...")
            time.sleep(10)
            embedding = get_doc_embedding(chunk)
            vector_db.append((chunk, embedding))

    return vector_db

# Example usage
vector_db = create_vector_db("path/to/your/text/file.txt")
print(vector_db[:5])  # Print the first 5 entries of the vector database