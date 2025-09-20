from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader  # Import DirectoryLoader for batch loading and PyPDFLoader for PDFs
import random  # Import random module to select a random document later

loader = DirectoryLoader(  # Create a DirectoryLoader to load multiple PDF files from a folder
    path="../raw-data/Books",  # Path to the folder containing PDF files
    glob="*.pdf",  # Load only files with .pdf extension
    loader_cls=PyPDFLoader,  # Use PyPDFLoader to process each PDF file
)

docs = loader.load()  # Load all PDFs into a list of document objects
num_docs = len(docs)  # Get the total number of loaded documents
rand_doc = random.randint(0, num_docs - 1)  # Pick a random document index
print("Number of docs loaded : ", num_docs)  # Print the total number of documents
print(f"Random document content: \n{docs[rand_doc].page_content}")  # Print the content of a random document
print(f"Metadata of the document: {docs[rand_doc].metadata}")  # Print metadata of the random document
