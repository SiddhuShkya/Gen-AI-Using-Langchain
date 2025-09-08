import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ------------------ Environment Setup ------------------ #

# Load environment variables (e.g., API keys) from a .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Retrieve OpenAI API key

# ------------------ Model Setup ------------------ #

# Initialize an OpenAI Chat model via LangChain
# Parameters:
# - model="gpt-4o" → Specifies which OpenAI model to use
# - temperature=0.5 → Controls randomness (0 = deterministic, 1 = more creative)
# - max_completion_tokens=15 → Restricts output length
# - api_key=api_key → Authenticates with OpenAI API
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    max_completion_tokens=15,
    api_key=api_key
)

# ------------------ Inference ------------------ #

# Run inference (send a prompt to the model)
result = model.invoke("What is the capital of Nepal?")

# Print the result
# Note: Will raise an error if OPENAI_API_KEY is invalid or if the model is not accessible
print(result)
print(result.content)
