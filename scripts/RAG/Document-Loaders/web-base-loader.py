import os  # For handling file paths
import sys  # To modify Python path
from langchain_community.document_loaders import WebBaseLoader  # Loader for web pages
from langchain_core.output_parsers import StrOutputParser    # Parses LLM output as string
from langchain.prompts import PromptTemplate  # Template for prompting LLMs

# Add project root to Python path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_gemma  # Custom function to load Gemma model

url = "https://httpbin.org/html"  # URL of the web page to load

# Define a prompt template to extract title/topic from web content
prompt = PromptTemplate(
    template="What is the below content about : \n {content} \n Just generate me the title/topic for the content",
    input_variables=['content']
)

model = load_gemma()  # Load the Gemma language model
parser = StrOutputParser()  # Initialize parser to convert LLM output to string
loader = WebBaseLoader(url)  # Initialize loader for the given URL
docs = loader.load()  # Load web page content into document objects

content = docs[0].page_content  # Extract content of the first loaded document

chain = prompt | model | parser  # Create a chain: prompt -> model -> parser
result = chain.invoke({'content': content})  # Run the chain on the web content
print(f"\nPredicted {result}")  # Print the predicted title/topic
