from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI # type: ignore
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


def load_gemma():
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        huggingfacehub_api_token=api_token,
        timeout=120
    )
    model = ChatHuggingFace(llm=llm)
    return model


def load_mistral():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        huggingfacehub_api_token=api_token,
        timeout=120
    )
    model = ChatHuggingFace(llm=llm)
    return model

    
def load_openai():
    model = ChatOpenAI()
    return model

def load_claude():
    model = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')
    return model