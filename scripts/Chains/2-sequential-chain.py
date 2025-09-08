import os
import sys
from dotenv import load_dotenv # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore

# Add parent directory (two levels up) to the Python path
# This allows importing modules from the 'utils' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the custom function to load the Mistral model
from utils.load_llms import load_mistral

# Load environment variables from a .env file
load_dotenv()

# Define the first prompt template to generate a detailed report
# {topic} will be replaced when invoking the chain
prompt1 = PromptTemplate(
    template='Genrate a detailed report on {topic}',  # Note: small typo "Genrate" should be "Generate"
    input_variables=['topic']
)

# Define the second prompt template to summarize text into 5 points
# {text} will be replaced with the output of the first prompt
prompt2 = PromptTemplate(
    template='Generate a 5 points summary from the following text \n {text}',
    input_variables=['text']
)

# Load the Mistral model
model = load_mistral()

# Define an output parser to convert model outputs to plain strings
parser = StrOutputParser()

# Chain the prompts, model, and parser
# Flow: prompt1 -> model -> parser -> model -> parser
# Essentially: generate a detailed report -> parse -> summarize -> parse
chain = prompt1 | model | parser | model | parser

# Invoke the chain with a specific topic
result = chain.invoke({'topic': 'Unemployment in Nepal'})

# Print the final output (summarized 5-point report)
print(result)

# Print an ASCII representation of the chain structure for visualization
chain.get_graph().print_ascii()
