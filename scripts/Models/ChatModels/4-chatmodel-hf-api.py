import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# ------------------ Environment Setup ------------------ #

# Load environment variables (e.g., Hugging Face API token) from .env file
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")  # Hugging Face API key

# ------------------ Model Setup ------------------ #

# Initialize Hugging Face LLM endpoint
# - repo_id → model repo on Hugging Face
# - task → specifies task type (here: text generation)
# - huggingfacehub_api_token → authenticate using API token
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=api_token
)

# Wrap the model in LangChain's Chat interface
model = ChatHuggingFace(llm=llm)

# ------------------ Inference ------------------ #

# Send a query to the model (example usage)
result = model.invoke("What is the capital of Australia?")

# Print only the text content (instead of full metadata object)
print(result.content)
