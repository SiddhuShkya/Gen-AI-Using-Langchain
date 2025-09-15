import os
import sys
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

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

joke_gen_chain = RunnableSequence(initial_prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(explain_output_prompt, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'Football'})
print(result['joke'])
print(result['explanation'])