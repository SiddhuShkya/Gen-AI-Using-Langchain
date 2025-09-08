import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o", temperature=0.5, max_completion_tokens=15, api_key=api_key)
result = model.invoke("What is the capital of Nepal?")  
print(result)  # Output the result (Will throw an error because the model is not open sourced)   




 
