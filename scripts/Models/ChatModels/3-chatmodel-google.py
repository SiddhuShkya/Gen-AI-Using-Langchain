from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# ------------------ Environment Setup ------------------ #

# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")  # Retrieve Google API key

# ------------------ Model Setup ------------------ #

# Initialize Google's Gemini model via LangChain
# Parameters:
# - model="gemini-1.5-pro" → Specifies Gemini model version
# - temperature=0.5 → Controls randomness (lower = deterministic, higher = creative)
# - api_key=api_key → Authenticate with Google Generative AI API
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    api_key=api_key
)

# ------------------ Inference ------------------ #

# Send a prompt to the model (example query)
result = model.invoke("What is the capital of India?")

# Print the result
# Note: result is a ChatMessage object, so use `.content` to get plain text
print(result.content)
