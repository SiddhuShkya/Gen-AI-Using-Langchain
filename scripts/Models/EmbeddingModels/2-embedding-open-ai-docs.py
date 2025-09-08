from langchain_openai import OpenAIEmbeddings # type: ignore
from dotenv import load_dotenv # type: ignore
import os     

# Load environment variables from .env file
load_dotenv()   
api_token = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_token,
    dimensions=32
) 

documents = [
    "Kathmandu is the capital of Nepal.",
    "Mount Everest is the highest mountain in the world.",
    "The capital of France is Paris.",
    "The Great Wall of China is visible from space.",
    "The Nile is the longest river in the world."
]

# result = embedding.embed_query("Kathmandu is the capital of nepal")  # Example usage
result = embedding.embed_documents(documents)
print(result)