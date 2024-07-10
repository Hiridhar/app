# -*- coding: utf-8 -*-
"""Llama_3.py"""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Llama 3 Chatbot", layout="wide")

# Load the model and tokenizer
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
)

"""## Build a Chat App"""

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
    ]

def generate_response(messages):
    # Construct the prompt from messages
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "System: " + message["content"] + "\n"
        elif message["role"] == "user":
            prompt += "User: " + message["content"] + "\n"
        elif message["role"] == "assistant":
            prompt += "Assistant: " + message["content"] + "\n"

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9)
    response_msg = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_msg[len(prompt):].strip()

st.title("Llama 3 Chatbot")

# Function to handle user input
def handle_input():
    user_input = st.chat_input("Enter your message:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = generate_response(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
handle_input()
