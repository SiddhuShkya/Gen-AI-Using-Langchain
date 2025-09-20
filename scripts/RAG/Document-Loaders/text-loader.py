import os  # For handling file paths
import sys  # To modify Python path
from langchain_community.document_loaders import TextLoader  # Loader for text documents
from langchain_core.output_parsers import StrOutputParser    # Parses LLM output as string
from langchain_core.prompts import PromptTemplate            # Template for prompting LLMs

# Add project root to Python path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_mistral  # Custom function to load Mistral model

model = load_mistral()  # Load the Mistral language model

# Define a prompt template to extract title/topic from content
prompt = PromptTemplate(
    template="What is the below content about : \n {content} \n Just generate me the title/topic for the content",
    input_variables=['content']
)

parser = StrOutputParser()  # Initialize a parser to get LLM output as a string

loader = TextLoader("../raw-data/data.txt")  # Load text documents from a file
docs = loader.load()  # Read the documents into memory

print(f"Loaded {len(docs)} document(s).")  # Show number of documents loaded
print(f"First document content (truncated): {docs[0].page_content}")  # Print first document's content
print(f"Metadata of first document: {docs[0].metadata}")  # Print metadata of first document

chain = prompt | model | parser  # Create a chain: prompt -> model -> parser

result = chain.invoke({'content': docs[0].page_content})  # Run the chain on the first document

print(f"\nPredicted {result}")  # Print the predicted title/topic
