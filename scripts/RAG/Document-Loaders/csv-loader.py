from langchain_community.document_loaders import CSVLoader  # Import CSVLoader to load CSV files as documents

loader = CSVLoader(file_path="../raw-data/data.csv")  # Create a CSVLoader instance with the path to the CSV file

docs = loader.load()  # Load the CSV data into document objects
print(f"First document content: \n{docs[0].page_content}")  # Print the content of the first document
print(f"Metadata of first document: {docs[0].metadata}")  # Print metadata of the first document
print(f"Loaded {len(docs)} document(s).")  # Print the total number of loaded documents
