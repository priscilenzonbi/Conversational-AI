import streamlit as st
import os
from dotenv import load_dotenv
from google import genai

# Read the key from the .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.title("Chatbot 1.0")

# Creating the client once and reuse it
@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)

client = get_client()

# The conversation lives in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box at the bottom
user_text = st.chat_input("Say something...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # ENGINE SWAP: build the conversation text and send it to Gemini
    conversation = ""
    for m in st.session_state.messages:
        conversation += f"{m['role']}: {m['content']}\n"

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=genai.types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Keep answers clear and concise, no more than a short paragraph unless asked for detail."
        ),
        contents=conversation
    )
    reply = response.text
   # reply = response.text

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)