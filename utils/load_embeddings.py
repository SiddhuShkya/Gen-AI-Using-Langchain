import os
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------ Environment Setup ------------------ #

# Load environment variables (like API keys) from a .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")  # Hugging Face access token

# Authenticate with Hugging Face Hub using the token
if hf_token:
    login(token=hf_token)
else:
    # Raise an error if the token is missing
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file")


# ------------------ Embedding Loaders ------------------ #

def load_embedding_mini():
    """
    Load a lightweight embedding model (MiniLM-L6-v2) 
    via LangChain's HuggingFaceEmbeddings wrapper.
    
    - Produces 384-dimensional embeddings.
    - Optimized for semantic search, clustering, and retrieval.
    """
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return model


def load_embedding_gemma():
    """
    Load Google's EmbeddingGemma (300M) using SentenceTransformers directly.
    
    - Produces high-quality embeddings from Gemma.
    - Typically larger and more resource-heavy than MiniLM.
    - Good for tasks needing richer semantic representations.
    """
    model = SentenceTransformer("google/embeddinggemma-300m")
    return model
