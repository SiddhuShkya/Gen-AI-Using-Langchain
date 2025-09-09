import os 
import sys
from langchain_core.prompts import load_prompt
import streamlit as st
  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.load_llms import load_gemma

model = load_gemma()

st.header("Research Tool")

paper_input = st.selectbox(
    "Select a title to describe",
    ["Paper 1: AI in Healthcare", "Paper 2: Quantum Computing Advances", "Paper 3: Renewable Energy Innovations"]
)

style_input = st.selectbox(
    "Select a style for the summary",
    ["Concise", "Detailed", "Technical"]   
)

length_input = st.selectbox(
    "Select the length of the summary",
    ["Short (1 paragraph)", "Medium (2 paragraph)", "Long (3+ paragraph)"]
)

#template 
template = load_prompt("template.json")

prompt = template.invoke(
    {
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input':length_input 
    }
)

if st.button('Summarize'): 
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input':length_input
    })
    # st.write(f"Summarizing {paper_input} in {style_input} style with {length_input} length...")
    # result = model.invoke(prompt)  
    reply = result.content.split("<|assistant|>")[-1].strip()
    st.write(reply)
    


# result = model.invoke("Who is albert einstein?")  # Example usage
# # Extract only the assistant's reply
# reply = result.content.split("<|assistant|>")[-1].strip()
# print(reply)
