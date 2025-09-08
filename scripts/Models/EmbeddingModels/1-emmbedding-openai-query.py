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

# ------------------ Generate Embedding ------------------ #

# Embed a text query into a numerical vector
result = embedding.embed_query("Kathmandu is the capital of nepal")  # Example usage

# Print the resulting embedding vector
print(str(result))
