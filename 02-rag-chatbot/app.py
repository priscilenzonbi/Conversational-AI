import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.title("Retrieval-augmented chatbot")

@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)

client = get_client()

# 1. THE KNOWLEDGE BASE: the only facts the bot is allowed to use
documents = [
    "Transfers between PayFlow accounts are free. Transfers to external banks cost 1.5% of the amount, capped at 5 dollars.",
    "A PayFlow account can be opened by anyone aged 18 or older with a valid national ID. There is no minimum opening balance.",
    "If your card is lost or stolen, freeze it instantly in the app under Cards then Freeze. A replacement card costs 3 dollars and arrives in 5 working days.",
    "PayFlow does not offer loans or credit. We provide current accounts, savings pots, and card services only.",
    "Customer support is available 24/7 through in-app chat. Phone support runs 8am to 8pm, Monday to Friday.",
]

# 2. EMBED every document once, turning each into a meaning-vector
@st.cache_resource
def embed_documents():
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=documents,
    )
    return [np.array(e.values) for e in result.embeddings]

doc_vectors = embed_documents()

# 3. COSINE SIMILARITY: a number for how close two vectors are in meaning
def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 4. RETRIEVE the most relevant documents for a question
def retrieve(question, top_k=2):
    q = client.models.embed_content(model="gemini-embedding-001", contents=question)
    q_vector = np.array(q.embeddings[0].values)
    scores = [similarity(q_vector, dv) for dv in doc_vectors]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_text = st.chat_input("Ask about PayFlow...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Retrieve relevant facts, then ground the model in them
    retrieved = retrieve(user_text, top_k=2)
    context = "\n".join(retrieved)

    prompt = f"""Answer the question using ONLY the information below.
If the answer is not in the information, say you do not have that information.

Information:
{context}

Question: {user_text}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="You are a concise PayFlow support assistant."
        ),
        contents=prompt,
    )
    reply = response.text

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
        with st.expander("Sources used"):
            for r in retrieved:
                st.write(f"- {r}")