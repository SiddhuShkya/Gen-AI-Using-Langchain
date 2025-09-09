import os
import sys
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.load_llms import load_gemma

model = load_gemma()

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    result = model.invoke(chat_history)  # Example usage
    reply = result.content.split("<|assistant|>")[-1].strip()
    chat_history.append(AIMessage(content=reply))
    print(f"Bot: {reply}")
    
print(chat_history)