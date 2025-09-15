import os
import sys
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_gemma


initial_prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

explain_output_prompt = PromptTemplate(
    template='Explain the following joke -\n {text}',
    input_variables=['text']
)

model = load_gemma()
parser = StrOutputParser()

chain = RunnableSequence(initial_prompt, model, parser, explain_output_prompt, model, parser)

result = chain.invoke({'topic': 'AI'})
print(result)
