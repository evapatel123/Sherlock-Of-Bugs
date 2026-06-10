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
    }
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
    }
]

def response(message,history):
    messages=[{"role":"system","content":"You are a debugger."}]

    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(messages,max_tokens=1000)
    
    return response.choices[0].message.content.strip() 
    
chatbot= gr.ChatInterface(response,title="🕵️‍♂️ The Sherlock of Bugs",description="Your personal debugging partner from a crime scene to clean code.")
chatbot.launch()
