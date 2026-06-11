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
        "context": "Occurs when trying to access an index that doesn't exist in a list. Remember Python and many other programming languages lists start at index 0. If a list has 3 items, valid indices are 0, 1, and 2."
    },
    {
        "error_type": "TypeError: 'int' object is not iterable",
        "context": "Occurs when trying to loop through or unpack an integer instead of a collection like a list or dictionary. Check if you passed a number to a for loop or any other type of loop."
    },
    {
        "error_type": "KeyError",
        "context": "Occurs when trying to access a dictionary key that does not exist. Check for typos or use the .get() method to handle missing keys safely."
    },
    {
        "error_type": "IndentationError: unexpected indent",
        "context": "Python and many other programming languages relies heavily on whitespace. Some languages like C++ and Javascript may rely on curly braces. Ensure you are consistently using either 4 spaces or 1 tab inside loops and functions."
    },
    {
        "error_type": "NameError: name is not defined",
        "context": "Occurs when using a variable or function name that Python or any other programming language doesn't recognize. Common causes: a typo, or calling it before it is defined."
    },
    {
        "error_type": "TypeError: unsupported operand type(s)",
        "context": "Occurs when performing an operation between incompatible data types, such as adding a string and an integer. Check the types of the variables involved."
    },
    {
        "error_type": "ValueError",
        "context": "Occurs when a function receives the correct type of argument but an invalid value. Example: converting 'hello' to an integer using int()."
    },
    {
        "error_type": "AttributeError",
        "context": "Occurs when trying to access a method or attribute that an object does not have. Check for typos or whether the object is the expected type."
    },
    {
        "error_type": "ZeroDivisionError",
        "context": "Occurs when attempting to divide a number by zero. Ensure the denominator is not zero before performing division."
    },
    {
        "error_type": "SyntaxError",
        "context": "Occurs when Python encounters invalid code structure. Common causes include missing colons, parentheses, quotes, or incorrect statement formatting."
    },
    {
        "error_type": "ModuleNotFoundError",
        "context": "Occurs when Python cannot find the module being imported. Verify the module is installed and the import statement is correct."
    },
    {
        "error_type": "ImportError",
        "context": "Occurs when Python finds a module but cannot import a specific object from it. Check spelling and module contents."
    },
    {
        "error_type": "FileNotFoundError",
        "context": "Occurs when trying to open or access a file that does not exist at the specified path. Verify the filename and location."
    },
    {
        "error_type": "PermissionError",
        "context": "Occurs when the program does not have permission to access a file, folder, or system resource. Check file permissions and access rights."
    },
    {
        "error_type": "UnboundLocalError",
        "context": "Occurs when a local variable is referenced before it has been assigned a value within a function."
    },
    {
        "error_type": "RecursionError",
        "context": "Occurs when a function calls itself too many times without reaching a stopping condition. Verify the base case of recursive functions."
    },
    {
        "error_type": "MemoryError",
        "context": "Occurs when the program runs out of available memory. This often happens with very large datasets or infinite data structures."
    },
    {
        "error_type": "AssertionError",
        "context": "Occurs when an assert statement evaluates to False. Check whether the expected condition is actually true."
    },
    {
        "error_type": "RuntimeError",
        "context": "A generic error indicating something unexpected happened during execution that doesn't fit a more specific exception type."
    },
    {
        "error_type": "KeyboardInterrupt",
        "context": "Occurs when the user manually stops the program, usually by pressing Ctrl+C."
    },
    {
        "error_type": "EOFError",
        "context": "Occurs when input() reaches the end of input unexpectedly. Common in online judges or automated testing environments."
    },
    {
        "error_type": "Wrong Data Type",
        "context": "The program runs but produces incorrect results because a variable contains a different type than expected."
    }
]
kb_contexts = []
kb_embeddings = []

for item in knowledge_base:
    context_text = item["context"]
    kb_contexts.append(context_text)
    

    embedding = encoder_model.encode(context_text)
    kb_embeddings.append(embedding)

kb_embeddings = np.array(kb_embeddings)

def get_relevant_context(user_message):
    #Finds the most relevant error context from the knowledge base using cosine similarity.
    user_embedding = encoder_model.encode(user_message)
    
    # Calculate cosine similarities
    # I searched up on Google how to use the linalg.norm() function and the .argmax() function from numpy, becuase it was giving me an error initially.
    #The whole formula calculates the angle between them. If the angle is very small, it means the two sentences have a very similar meaning.
    #It gives back a score between 0 (totally different) and 1 (exactly the same).
    
    similarities = np.dot(kb_embeddings, user_embedding) / (
        np.linalg.norm(kb_embeddings, axis=1) * np.linalg.norm(user_embedding)
    )

    # This part find the best index at which the similarity is the most, aka finds the highest number in the similarities vector.
    
    best_idx = np.argmax(similarities)

    if similarities[best_idx] > 0.3:
        return kb_contexts[best_idx]

def response(message,history):
    
    retrieved_context = get_relevant_context(message)

    system_prompt = (
        "You are 'The Sherlock of Bugs', an interactive, clever, and supportive debugging mentor.\n\n"
        "YOUR RULES:\n"
        "1. When the user presents an error message, DO NOT give away the direct solution or fix immediately.\n"
        "2. Instead, use the provided Context to give them architectural clues or structural hints. Ask them how they think they should fix it, and encourage them to try writing a solution.\n"
        "3. If the user responds with a fix or an explanation, evaluate it. Tell them clearly if they are correct or incorrect.\n"
        "4. If they are correct, praise them and summarize the clean code. If they are struggling or incorrect, guide them closer or finally reveal the exact, elegant solution.\n\n"
        f"RELEVANT BACKGROUND CONTEXT FOR THIS CONVERSATION:\n{retrieved_context}"
    )
    messages=[{"role":"system","content": system_prompt}]

    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(messages,max_tokens=1000)
    
    return response.choices[0].message.content.strip() 


custom_css = """
body, .gradio-container { 
    background-color: #0b001a !important; 
    color: #ff77ff !important; 
    font-family: 'Segoe UI', system-ui, sans-serif !important; 
}
.gradio-container { 
    border: 2px solid #ff007f !important; 
    border-radius: 16px !important; 
    box-shadow: 0 0 35px rgba(255, 0, 127, 0.25), inset 0 0 15px rgba(112, 0, 255, 0.15) !important; 
    padding: 25px !important; 
    background: radial-gradient(circle at center, #1a0033 0%, #060010 100%) !important; 
}

/* Glowing Header Board */
.logo-header { 
    text-align: center; 
    border: 2px dashed #ff007f; 
    background: rgba(255, 0, 127, 0.03); 
    padding: 20px; 
    border-radius: 8px; 
    margin-bottom: 25px; 
    box-shadow: 0 0 20px rgba(255, 0, 127, 0.15); 
    animation: pulse-glow 4s infinite alternate; 
}
.logo-header h1 { 
    color: #ff007f !important; 
    font-size: 2.5rem; 
    letter-spacing: 4px; 
    font-weight: 900; 
    margin: 0; 
    text-shadow: 0 0 15px #ff007f, 0 0 30px #7000ff; 
}
.logo-header p { 
    color: #9d4edd; 
    margin-top: 8px; 
    font-size: 1rem; 
    letter-spacing: 2px; 
    text-transform: uppercase; 
    text-shadow: 0 0 5px #9d4edd; 
}

/* Chatbot Window Customization */
.chatbot {
    border: 1px solid #7000ff !important;
    background: rgba(15, 0, 30, 0.6) !important;
}

/* Styling Chat Bubbles Natively */
.message.user { 
    background-color: #3c006b !important; 
    color: #ffffff !important; 
    border: 1px solid #ff007f !important; 
    box-shadow: 0 0 10px rgba(255, 0, 127, 0.2);
}
.message.bot { 
    background-color: #16002c !important; 
    color: #ff77ff !important; 
    border: 1px solid #7000ff !important; 
    box-shadow: 0 0 10px rgba(112, 0, 255, 0.2);
}

/* Buttons Styling */
.cyber-btn { 
    background: #ff007f !important; 
    color: #ffffff !important; 
    font-weight: bold !important; 
    border: none !important; 
    border-radius: 6px !important; 
    box-shadow: 0 0 15px rgba(255, 0, 127, 0.4) !important; 
    cursor: pointer; 
    transition: all 0.25s ease !important; 
    text-transform: uppercase; 
    letter-spacing: 1px; 
}
.cyber-btn:hover { 
    background: #7000ff !important; 
    box-shadow: 0 0 25px #7000ff !important; 
}

.secondary-btn { 
    background-color: rgba(112, 0, 255, 0.1) !important; 
    color: #ff77ff !important; 
    border: 1px solid #7000ff !important; 
    border-radius: 6px !important; 
    box-shadow: 0 0 10px rgba(112, 0, 255, 0.2) !important; 
    text-transform: uppercase; 
    transition: all 0.25s ease !important;
}
.secondary-btn:hover { 
    background: rgba(255, 0, 127, 0.15) !important; 
    border-color: #ff007f !important;
    box-shadow: 0 0 20px #ff007f !important; 
    color: #fff !important;
}

footer { display: none !important; }

@keyframes pulse-glow {
    0% { box-shadow: 0 0 15px rgba(255, 0, 127, 0.1); border-color: #ff007f; }
    100% { box-shadow: 0 0 30px rgba(112, 0, 255, 0.4); border-color: #7000ff; }
}
"""
    
chatbot= gr.ChatInterface(response,title="🕵️‍♂️ The Sherlock of Bugs",description="Your personal debugging partner from a crime scene to clean code.")
chatbot.launch()