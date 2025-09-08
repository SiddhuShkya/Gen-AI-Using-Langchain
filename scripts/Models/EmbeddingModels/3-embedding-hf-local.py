from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Kathmandu is the capital of Nepal."
documents = [
    "Kathmandu is the capital of Nepal.",
    "Mount Everest is the highest mountain in the world.",
    "The capital of France is Paris.",
    "The Great Wall of China is visible from space.",
    "The Nile is the longest river in the world."
]
# vector = embedding.embed_query(text)  # Example usage
vector = embedding.embed_documents(documents)  # Example usage
# Extract the first vector from the list of vector
print(str(vector))  # Print the embedding vector