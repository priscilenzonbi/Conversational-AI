import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.title("Chat With Your Document")

@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)

client = get_client()

# --- Helper functions ---

def read_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "".join((page.extract_text() or "") for page in reader.pages)
    else:  # plain text
        return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_text(text, chunk_size=1500, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def embed_texts(texts):
    result = client.models.embed_content(model="gemini-embedding-001", contents=texts)
    return [np.array(e.values) for e in result.embeddings]

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(question, top_k=3, threshold=0.5):
    q = client.models.embed_content(model="gemini-embedding-001", contents=question)
    q_vec = np.array(q.embeddings[0].values)
    scores = [similarity(q_vec, v) for v in st.session_state.chunk_vectors]
    order = np.argsort(scores)[::-1][:top_k]
    return [(st.session_state.chunks[i], scores[i]) for i in order if scores[i] >= threshold]

# --- Upload phase ---
uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded is not None and st.session_state.get("current_file") != uploaded.name:
    with st.spinner("Reading and embedding your document..."):
        text = read_file(uploaded)
        chunks = chunk_text(text)
        st.session_state.chunks = chunks
        st.session_state.chunk_vectors = embed_texts(chunks)
        st.session_state.current_file = uploaded.name
        st.session_state.messages = []
    st.success(f"Loaded {uploaded.name} into {len(chunks)} chunks. Ask away.")

# --- Chat phase ---
if "chunks" in st.session_state:
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_text = st.chat_input("Ask about your document...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        relevant = retrieve(user_text)

        if not relevant:
            reply = "I could not find anything relevant to that in the uploaded document."
            with st.chat_message("assistant"):
                st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            context = "\n\n".join(c for c, s in relevant)
            prompt = f"""Answer using ONLY the information below.
If the answer is not in the information, say you do not have that information.

Information:
{context}

Question: {user_text}"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction="You answer questions about the user's uploaded document. Be concise and accurate."
                ),
                contents=prompt,
            )
            reply = response.text
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)
                with st.expander("Sources used (with similarity scores)"):
                    for c, s in relevant:
                        st.write(f"**score {s:.2f}** — {c[:200]}...")
else:
    st.info("Upload a document above to start chatting.")