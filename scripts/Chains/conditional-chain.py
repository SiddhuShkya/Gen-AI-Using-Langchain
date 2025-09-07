import os
import sys
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda # type: ignore
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

# Add parent directory (three levels up) to Python path
# This allows importing custom modules from 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import a custom function to load the OpenAI LLM
from utils.load_llms import load_openai

# Load environment variables from a .env file
load_dotenv()

# Define a Pydantic model for classifying feedback sentiment
class FeedBack(BaseModel):
    # Only 'positive' or 'negative' are valid sentiments
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Load the OpenAI model
model = load_openai()

# Create a Pydantic output parser to enforce structured output according to FeedBack model
parser = PydanticOutputParser(pydantic_object=FeedBack)

# Prompt template to classify sentiment of feedback
# Includes format instructions from the parser to ensure structured output
prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative. \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Prompt template for responding to positive feedback
prompt_pos = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Prompt template for responding to negative feedback
prompt_neg = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Chain to classify sentiment using the model and parser
classifier_chain = prompt | model | parser

# Branch chain to respond according to sentiment
# - If sentiment is positive -> use prompt_pos
# - If sentiment is negative -> use prompt_neg
# - Otherwise, return default message
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'] == 'positive', prompt_pos | model | parser),
    (lambda x: x['sentiment'] == 'negative', prompt_neg | model | parser),
    RunnableLambda(lambda x: "Could not find the sentiment")
)

# Combine classifier and branch chains into a single pipeline
chain = classifier_chain | branch_chain

# Example feedback to classify and respond to
feedback = "This is a terrible smart phone"

# Run the full chain
# Flow:
#   1. Classify sentiment
#   2. Branch to appropriate response prompt based on sentiment
result = chain.invoke({"feedback": feedback})
# Print the sentiment from the result
# result is expected to conform to FeedBack model
print(result)
