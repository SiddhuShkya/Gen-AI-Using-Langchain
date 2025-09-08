from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, api_key=api_key)
result = model.invoke("What is the capital of India?")  # Example usage
print(result)