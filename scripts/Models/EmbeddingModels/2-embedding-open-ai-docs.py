from langchain_openai import OpenAIEmbeddings  # LangChain wrapper for OpenAI embeddings
from dotenv import load_dotenv  # Load environment variables from .env
import os     

# ------------------ Environment Setup ------------------ #

# Load environment variables (like OpenAI API key) from a .env file
load_dotenv()   
api_token = os.getenv("OPENAI_API_KEY")  # Retrieve OpenAI API key

# ------------------ Embedding Model Setup ------------------ #

# Initialize OpenAI embedding model via LangChain
# Parameters:
# - model → which embedding model to use ("text-embedding-3-small")
# - openai_api_key → authentication with OpenAI API
# - dimensions → optional; specifies embedding dimension (32 in this example, usually 1536 for text-embedding-3-small)
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_token,
    dimensions=32
) 

# ------------------ Example Documents ------------------ #

documents = [
    "Kathmandu is the capital of Nepal.",
    "Mount Everest is the highest mountain in the world.",
    "The capital of France is Paris.",
    "The Great Wall of China is visible from space.",
    "The Nile is the longest river in the world."
]

# ------------------ Generate Embeddings ------------------ #

# Embed multiple documents into numerical vectors
# Returns a list of embedding vectors corresponding to each document
result = embedding.embed_documents(documents)

# Print the resulting embedding vectors
print(result)
