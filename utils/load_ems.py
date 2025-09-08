import os
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Authenticate with Hugging Face
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file")



def load_embedding_mini():
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return model


def load_embedding_gemma():
    model = SentenceTransformer("google/embeddinggemma-300m")
    return model