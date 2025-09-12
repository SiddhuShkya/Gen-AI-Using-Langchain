import os
import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_gemma

# Load the model
model = load_gemma()

# Prompt for detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# Prompt for 5-line summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=['text']
)

# String output parser
parser = StrOutputParser()

# Build chain: report → model → parse → summary → model → parse
chain = template1 | model | parser | template2 | model | parser

# Run chain with input
result = chain.invoke({'topic': 'black hole'})

# Print final result
print(result)
