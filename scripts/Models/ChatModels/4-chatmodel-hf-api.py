import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()        
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task ="text-generation",
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(
    llm=llm,
)

result = model.invoke("What is the capital of Australia?")  # Example usage
print(result)  