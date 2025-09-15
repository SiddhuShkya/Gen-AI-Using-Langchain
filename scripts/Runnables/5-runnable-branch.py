import os
import sys
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_gemma



prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template="Summarize the following text\n{text}",
    input_variables=['text'] 
)

model = load_gemma()
parser = StrOutputParser()

reoprt_gen_chain = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(reoprt_gen_chain, branch_chain)
result = final_chain.invoke({'topic': 'Russia vs Ukraine'})
print(result)