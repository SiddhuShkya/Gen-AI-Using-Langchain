import os
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

# ------------------ Environment Setup ------------------ #

# Load environment variables from a .env file (like Hugging Face API token)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Authenticate with Hugging Face Hub
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file")

# ------------------ Load Model ------------------ #

# Download and load the SentenceTransformer model from Hugging Face
# "google/embeddinggemma-300m" is a pretrained embedding model
model = SentenceTransformer("google/embeddinggemma-300m")

# ------------------ Example Query and Documents ------------------ #

query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# ------------------ Generate Embeddings ------------------ #

# Encode the query into a vector
query_embeddings = model.encode_query(query)

# Encode all documents into vectors
document_embeddings = model.encode_document(documents)

# Print the shapes of the embeddings
# query_embeddings.shape → (768,) for a single query
# document_embeddings.shape → (4, 768) for 4 documents
print(query_embeddings.shape, document_embeddings.shape)

# ------------------ Compute Similarities ------------------ #

# Compute cosine similarities between the query and each document
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# Example output: tensor([[0.3011, 0.6359, 0.4930, 0.4889]])

# ------------------ Rank Documents ------------------ #

# Convert similarities to a ranking (descending order)
# The highest similarity comes first
ranking = similarities.argsort(descending=True)[0]
print(ranking)
# Example output: tensor([1, 2, 3, 0]) → 1st document is most similar
