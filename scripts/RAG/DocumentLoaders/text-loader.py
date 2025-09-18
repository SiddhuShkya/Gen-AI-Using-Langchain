import os
import sys
from langchain_community.document_loaders import TextLoader  # Loader for text documents
from langchain_core.output_parsers import StrOutputParser    # Parses LLM output as string
from langchain_core.prompts import PromptTemplate            # Template for prompting LLMs

# Add project root to Python path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_mistral  # Custom function to load Mistral model

# Load the Mistral model
model = load_mistral()

# Define a prompt template to ask for the title of the content
prompt = PromptTemplate(
    template="What is the below content about : \n {content} \n Just generate me the title/topic for the content",
    input_variables=['content']
)

# Initialize string output parser
parser = StrOutputParser()

# Load documents from text file (relative path)
loader = TextLoader("../raw-data/data.txt")
docs = loader.load()

# Print document info concisely
print(f"Loaded {len(docs)} document(s).")
print(f"First document content (truncated): {docs[0].page_content}")  # Show only first 200 chars
print(f"Metadata of first document: {docs[0].metadata}")

# Create a chain: prompt -> model -> parser
chain = prompt | model | parser

# Run the chain on the first document
result = chain.invoke({'content': docs[0].page_content})

# Print the result
print(f"\nPredicted {result}")
