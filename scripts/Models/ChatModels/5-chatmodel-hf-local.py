from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline



load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "device_map": "cpu"
    },
)

model = ChatHuggingFace(
    llm=llm,
)

result = model.invoke("Who is albert einstein?")  # Example usage
# Extract only the assistant's reply
reply = result.content.split("<|assistant|>")[-1].strip()
print(reply)
