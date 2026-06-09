from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import gradio as gr
import numpy as np 

client=InferenceClient("Qwen/Qwen2.5-7B-Instruct")

def response(message,history):
    messages=[{"role":"system","content":"You are a friendly chatbot."}]

    if history:
        messages.extend(history)
    messages.append({"role":"user","content":message})

    response=client.chat_completion(messages,max_tokens=1000)
    
    return response.choices[0].message.content.strip() 
    
chatbot=gr.ChatInterface(response)
chatbot.launch()