import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.5, api_key=os.getenv("ANTHROPIC_API_KEY"))
result = model.invoke("What is the capital of Japan?")
print(result)  # Output the result (Will throw an error because the model is not open