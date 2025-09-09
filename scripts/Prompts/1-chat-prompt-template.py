
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You are a {domain} addict."),
    ('human', "Recommed me a {domain} based on {topic}.")
])

prompt = chat_template.invoke({
    'domain': 'movie',
    'topic': 'science fiction'
})

print(prompt)