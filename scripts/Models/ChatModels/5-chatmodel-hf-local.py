from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# ------------------ Environment Setup ------------------ #

# Load environment variables from a .env file (if needed for local models)
load_dotenv()

# ------------------ Model Setup ------------------ #

# Load a local Hugging Face model using HuggingFacePipeline
# Parameters:
# - model_id → specifies which model to load locally
# - task → task type (text generation)
# - model_kwargs → options for memory optimization and device mapping
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "torch_dtype": "auto",          # Automatically choose data type
        "low_cpu_mem_usage": True,      # Reduce memory usage on CPU
        "device_map": "cpu"             # Run model on CPU (change to "auto"/"cuda" for GPU)
    },
)

# Wrap the pipeline model with LangChain's chat interface
model = ChatHuggingFace(llm=llm)

# ------------------ Inference ------------------ #

# Send a prompt to the model
result = model.invoke("Who is albert einstein?")  # Example usage

# ------------------ Extract Assistant's Reply ------------------ #
# The output may include special tokens like <|assistant|>
# Split by <|assistant|> and take the last part to get clean text
reply = result.content.split("<|assistant|>")[-1].strip()

# Print the assistant's reply
print(reply)
