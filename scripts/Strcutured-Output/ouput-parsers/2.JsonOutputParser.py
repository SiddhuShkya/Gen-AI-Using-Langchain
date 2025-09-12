import os
import sys
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_gemma

# Load the model
model = load_gemma()

# JSON output parser
parser = JsonOutputParser()

# Prompt template for 5 facts in JSON format
template = PromptTemplate(
    template=(
        "Give exactly 5 facts about {topic}. "
        "Return the output strictly as a JSON array of objects, each having keys 'fact' and 'category'. "
        "Do not include any text outside JSON. "
        "Do not include markdown formatting like ```json. "
        "If a string contains quotes, escape them using \\\". "
        "{format_instruction}"
    ),
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Build chain: template → model → JSON parser
chain = template | model | parser

# Run chain with input
result = chain.invoke({'topic': 'The Shawshank Redemption'})

# Print final JSON result
print(result)
