from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import gradio as gr
import numpy as np 

encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")


# Hey guys! So i made this knowledge base and what this does is that it talks about the specific type of error that the chatbot might run into, and the explaination of what 
# the error message means, so that it could guide the user. 

knowledge_base = [
    {
        "error_type": "IndexError: list index out of range",
        "context": "Occurs when trying to access an index that doesn't exist in a list. Remember Python lists start at index 0. If a list has 3 items, valid indices are 0, 1, and 2."
    },
    {
        "error_type": "TypeError: 'int' object is not iterable",
        "context": "Occurs when trying to loop through or unpack an integer instead of a collection like a list or dictionary. Check if you passed a number to a for loop."
    },
    {
        "error_type": "KeyError",
        "context": "Occurs when trying to access a dictionary key that does not exist. Check for typos or use the .get() method to handle missing keys safely."
    },
    {
        "error_type": "IndentationError: unexpected indent",
        "context": "Python relies heavily on whitespace. Ensure you are consistently using either 4 spaces or 1 tab inside loops and functions."
    },
    {
        "error_type": "NameError: name is not defined",
        "context": "Occurs when using a variable or function name that Python doesn't recognize. Common causes: a typo, or calling it before it is defined."
    }
]

def response(message,history):
    messages=[{"role":"system","content":"You are a debugger."}]

    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(messages,max_tokens=1000)
    
    return response.choices[0].message.content.strip() 
    
chatbot= gr.ChatInterface(response)
chatbot.launch()
