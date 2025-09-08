from langchain_huggingface import HuggingFaceEmbeddings

# ------------------ Embedding Model Setup ------------------ #

# Initialize a Hugging Face embedding model via LangChain
# - model_name → specifies the pretrained SentenceTransformer model to use
# MiniLM-L6-v2 is lightweight and optimized for semantic search
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ Example Text and Documents ------------------ #

text = "Kathmandu is the capital of Nepal."
documents = [
    "Kathmandu is the capital of Nepal.",
    "Mount Everest is the highest mountain in the world.",
    "The capital of France is Paris.",
    "The Great Wall of China is visible from space.",
    "The Nile is the longest river in the world."
]

# ------------------ Generate Embeddings ------------------ #

# Embed a single text query
# vector = embedding.embed_query(text)

# Embed multiple documents at once
# Returns a list of embedding vectors corresponding to each document
vector = embedding.embed_documents(documents)

# ------------------ Output ------------------ #

# Print the embedding vectors
# Each vector is a list of floats representing the semantic meaning
print(str(vector))
