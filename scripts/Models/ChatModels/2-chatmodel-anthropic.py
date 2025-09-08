import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# ------------------ Environment Setup ------------------ #

# Load environment variables (e.g., API keys) from a .env file
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")  # Retrieve Anthropic API key

# ------------------ Model Setup ------------------ #

# Initialize Anthropic's Claude model via LangChain
# Parameters:
# - model="claude-3-5-sonnet-20241022" → Claude version
# - temperature=0.5 → Controls randomness (lower = deterministic, higher = creative)
# - api_key=api_key → API key authentication
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.5,
    api_key=api_key
)

# ------------------ Inference ------------------ #

# Send a prompt to the model
result = model.invoke("What is the capital of Japan?")

# Print the result
# Note: Output is a ChatMessage object, so you should access `.content` for text
print(result.content)
