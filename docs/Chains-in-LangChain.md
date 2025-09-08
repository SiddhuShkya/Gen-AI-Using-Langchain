# 📑 Chains in LangChain

## 🔹 What Are Chains?
When building applications with Large Language Models (LLMs), a single prompt → response flow is often not enough.  
Applications usually require **multiple steps** that connect together.  

A **Chain** in LangChain is a way to **combine multiple components**—such as prompts, LLMs, output parsers, and more—into a single, reusable workflow. Basically chains in langchain helps to create a pipeline that helps us to control the workflow of our LLM based application using using minimal coding.

---

## 🔹 Why Chains?
- **Structured Workflows**: Instead of writing messy, step-by-step code, chains allow you to build modular pipelines.  
- **Reusability**: Define once, reuse across applications.  
- **Composability**: Easily combine prompts, models, tools, and parsers.  
- **Debuggability**: Inspect each step clearly.  

**Example:**  
A simple app may need:  
1. Take a question from a user.  
2. Format it into a prompt.  
3. Send it to the LLM.  
4. Parse and return the answer.  

This can be defined as a **Chain**.

---

## 🔹 Types of Chains
LangChain provides multiple kinds of chains:

1. **Simple Chains**  
   - A single LLM call with a prompt.  
   - Example: `LLMChain`

2. **Sequential Chains**  
   - Output of one step is passed to the next.  
   - Example: `SimpleSequentialChain`, `SequentialChain`

3. **Router Chains**  
   - Dynamically choose which chain to run based on input.  
   - Example: `MultiPromptChain`

4. **Custom Chains**  
   - You can build your own chain logic by extending the `Chain` class.

---

## 🔹 Example: Simple Chain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Define the LLM
llm = OpenAI(model="gpt-3.5-turbo")

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="What are some fun facts about {topic}?"
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run("space")
print(response)
