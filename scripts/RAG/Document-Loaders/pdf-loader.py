from langchain_community.document_loaders import PyPDFLoader  # Import PyPDFLoader to load PDF files as documents

loader = PyPDFLoader("../raw-data/MyLiteratureReviews.pdf")  # Create a loader for a single PDF file

docs = loader.load()  # Load the PDF into a list of document objects
print(f"First document content: \n{docs[0].page_content}")  # Print content of the first document
print(f"Metadata of first document: {docs[0].metadata}")  # Print metadata of the first document
print(f"Loaded {len(docs)} document(s).")  # Print total number of loaded documents
