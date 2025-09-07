import os
import sys
from dotenv import load_dotenv # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore

# Add the parent directory (two levels up) to the Python path
# This allows importing modules from the 'utils' folder without installing them as packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom functions to load different LLMs
from utils.load_llms import load_gemma
# Load environment variables from a .env file into the system environment
load_dotenv()

# Load the Gemma model using the custom loader function
model = load_gemma()

# Define a prompt template for the model
# {topic} is a placeholder that will be replaced when invoking the chain
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Define an output parser that converts the model output into a simple string
parser = StrOutputParser()

# Chain the prompt, model, and parser together
# This creates a runnable pipeline: Prompt -> Model -> Parser
chain = prompt | model | parser 

# Invoke the chain with a specific topic
result = chain.invoke({'topic': 'Large Language Models'})

# Print the resulting string from the model
print(result)

# Print an ASCII visualization of the chain's structure
chain.get_graph().print_ascii()
