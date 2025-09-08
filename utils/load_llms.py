# Import required classes for integrating different LLM providers
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_openai import ChatOpenAI  # OpenAI models
from langchain_anthropic import ChatAnthropic  # Anthropic Claude models
from dotenv import load_dotenv
import os

# Load environment variables from a .env file (e.g., API keys)
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")  # Hugging Face API token


# ---------------- Hugging Face Models ---------------- #

def load_llama():
    """
    Load TinyLlama (1.1B) from Hugging Face using API endpoint.
    """
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Model repo on Hugging Face
        task="text-generation",                        # Task type
        huggingfacehub_api_token=api_token             # Auth token
    )
    # Wrap the LLM with LangChain's ChatHuggingFace for chat-based usage
    model = ChatHuggingFace(llm=llm)
    return model


def load_gemma():
    """
    Load Gemma (2B instruction-tuned) from Hugging Face.
    Includes a timeout to handle longer requests.
    """
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        huggingfacehub_api_token=api_token,
        timeout=120
    )
    model = ChatHuggingFace(llm=llm)
    return model


def load_mistral():
    """
    Load Mistral (7B instruction-tuned) from Hugging Face.
    """
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        huggingfacehub_api_token=api_token,
        timeout=120
    )
    model = ChatHuggingFace(llm=llm)
    return model


def load_local_llm():
    """
    Load a local Hugging Face model (TinyLlama) using HuggingFacePipeline.
    This avoids API calls and runs inference locally on CPU.
    """
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        model_kwargs={
            "torch_dtype": "auto",          # Automatically select precision
            "low_cpu_mem_usage": True,      # Optimize memory usage
            "device_map": "cpu"             # Run on CPU (change to "auto" or "cuda" for GPU)
        },
    )
    model = ChatHuggingFace(llm=llm)
    return model


# ---------------- Other Providers ---------------- #

def load_openai():
    """
    Load OpenAI's ChatGPT model via LangChain.
    Requires OPENAI_API_KEY in environment variables.
    """
    model = ChatOpenAI()
    return model


def load_claude():
    """
    Load Anthropic's Claude model (Claude 3.7 Sonnet).
    Requires ANTHROPIC_API_KEY in environment variables.
    """
    model = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')
    return model
