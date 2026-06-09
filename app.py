from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import gradio as gr
import numpy as np 

client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

def response(message,history):
    messages=[{"role":"system","content":"You are a debugger."}]

    if history:
        messages.extend(history)
    messages.append({"role":"user","content":message})

    response=client.chat_completion(messages,max_tokens=1000)
    
    return response.choices[0].message.content.strip() 
    
chatbot=gr.ChatInterface(response)
chatbot.launch()