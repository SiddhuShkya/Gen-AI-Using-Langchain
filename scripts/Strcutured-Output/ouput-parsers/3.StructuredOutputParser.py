import os
import sys
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_gemma

# Load the model
model = load_gemma()

# Define response schema (3 facts)
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# Structured output parser with schema
parser = StructuredOutputParser.from_response_schemas(schema)

# Prompt template for 3 facts
template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Manual way (commented)
# prompt = template.invoke({'topic': 'Black Hole'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# Build chain: template → model → structured parser
chain = template | model | parser

# Run chain with input
result = chain.invoke({'topic': 'Blackhole'})

# Print final structured result
print(result)
