import re  # Import regex for text cleaning
from langchain.text_splitter import CharacterTextSplitter # Import text splitter
from langchain_community.document_loaders import PyPDFLoader # Import PDF loader

loader = PyPDFLoader("../raw-data/MyLiteratureReviews.pdf")  # Load PDF file
docs = loader.load()  # Read PDF into documents

# Clean up extra whitespace and newlines
for doc in docs:  
    doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()  # Replace multiple spaces/newlines with single space

splitter = CharacterTextSplitter(  # Create character-based text splitter
    separator = " ",  # Split by space
    chunk_size = 500,  # Max size of each chunk
    chunk_overlap  = 50,  # Overlap between chunks
)

chunks = splitter.split_documents(docs)  # Split documents into chunks

print(f"Total chunks: {len(chunks)}\n")  # Print total number of chunks
for i, chunk in enumerate(chunks, 1):  # Loop through chunks
    print(f"--- Chunk {i} ---")  # Print chunk number
    print(chunk)  # Print chunk content
