import os
import sys
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_llms import load_mistral, load_gemma

tweet_prompt = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

linkedin_prompt = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topics']
)

tweet_model = load_mistral()
linkedin_model = load_gemma()

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(tweet_prompt, tweet_model, parser),
    'linkedin': RunnableSequence(linkedin_prompt, linkedin_model, parser)
})

result = parallel_chain.invoke({'topic': 'AI'})
print("Generated tweet post : \n", result['tweet'])
print("\nGenerated linkedin post : \n", result['linkedin'])

