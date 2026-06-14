import streamlit as st
import os, glob, json, datetime
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

load_dotenv()
st.set_page_config(page_title="Faroty Support", page_icon="💬")
st.title("Faroty Support Assistant")

@st.cache_resource
def get_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client = get_client()
CONTACT_URL = "https://faroty.com/help/index.php?from=index&ticket=raise"

# ---------- Knowledge base ----------
def chunk_text(text, size=1200, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return [c for c in chunks if c.strip()]

@st.cache_resource
def build_knowledge():
    chunks, srcs = [], []
    for path in sorted(glob.glob("knowledge/*.txt")):
        with open(path, encoding="utf-8") as f:
            for c in chunk_text(f.read()):
                chunks.append(c); srcs.append(os.path.basename(path))
    if not chunks:
        return [], [], []
    res = client.models.embed_content(model="gemini-embedding-001", contents=chunks)
    return chunks, srcs, [np.array(e.values) for e in res.embeddings]

chunks, sources, vectors = build_knowledge()

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(question, top_k=3, threshold=0.5):
    q = client.models.embed_content(model="gemini-embedding-001", contents=question)
    qv = np.array(q.embeddings[0].values)
    scored = [(chunks[i], sources[i], similarity(qv, vectors[i])) for i in range(len(chunks))]
    scored.sort(key=lambda x: x[2], reverse=True)
    return [t for t in scored[:top_k] if t[2] >= threshold]

# ---------- Router ----------
def classify_intent(message):
    prompt = f"""You are the routing brain of a support bot for Faroty, a fintech app
for groups, associations, tontines, savings, and crowdfunding.
Classify the message into exactly one category:
- "general": a how-to or informational question documentation could answer.
- "account_specific": about the user's OWN account, money, a transaction, a dispute, or complaint. Needs a human.
- "human_request": the user explicitly asks for a person.
Also detect the language of the message: "fr" or "en".
Respond with JSON only: {{"category": "<category>", "language": "<fr or en>", "reason": "<short>"}}
User message: {message}"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        contents=prompt)
    try:
        return json.loads(resp.text)
    except Exception:
        return {"category": "general", "reason": "fallback"}

# ---------- Answering ----------
def answer_from_docs(question, retrieved):
    context = "\n\n".join(c for c, s, sc in retrieved)
    prompt = f"""Answer using ONLY the information below.
If the answer is not fully contained in it, say you are not sure and that a human can help.

Information:
{context}

Question: {question}"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a friendly, concise Faroty support assistant. Always reply in the same language the user wrote in, French or English. Never invent fees, rules, or steps."),
        contents=prompt)
    return resp.text

# ---------- Logging ----------
def log(entry):
    entry["timestamp"] = datetime.datetime.now().isoformat()
    with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

ESCALATION = {
    "en": f"This looks like something our support team should handle directly. You can reach a human agent here: {CONTACT_URL}",
    "fr": f"Cette demande doit être traitée directement par notre équipe d'assistance. Vous pouvez contacter un agent ici : {CONTACT_URL}",
}
NO_MATCH = {
    "en": f"I could not find this in our help articles. Our team can help you directly: {CONTACT_URL}",
    "fr": f"Je n'ai pas trouvé cette information dans nos articles d'aide. Notre équipe peut vous aider directement : {CONTACT_URL}",
}

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask about Faroty...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        intent = classify_intent(user_text)
        cat = intent["category"]
        lang = intent.get("language", "en")
        entry = {"question": user_text, "category": cat, "language": lang, "reason": intent.get("reason", "")}

        if cat in ("account_specific", "human_request"):
            reply = ESCALATION.get(lang, ESCALATION["en"])
            entry["action"] = "escalated"; entry["retrieved"] = []
        else:
            retrieved = retrieve(user_text)
            if not retrieved:
                reply = NO_MATCH.get(lang, NO_MATCH["en"])
                entry["action"] = "escalated_no_match"; entry["retrieved"] = []
            else:
                reply = answer_from_docs(user_text, retrieved)
                entry["action"] = "answered"
                entry["context"] = "\n\n".join(c for c, s, sc in retrieved)
                entry["retrieved"] = [{"source": s, "score": round(float(sc), 3)} for c, s, sc in retrieved]
        entry["answer"] = reply
        log(entry)

    except errors.ClientError as e:
        if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
            reply = ("We are handling a lot of requests right now. Please try again in a moment. / "
                     "Nous traitons beaucoup de demandes en ce moment. Veuillez réessayer dans un instant.")
        else:
            reply = ("Sorry, something went wrong. Please try again or contact support. / "
                     "Désolé, une erreur s'est produite. Veuillez réessayer ou contacter le support.")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
        
if st.session_state.messages:
    st.divider()
    st.caption("How helpful was this conversation? / Cette conversation vous a-t-elle aidé ?")
    rating = st.feedback("stars", key="session_rating")
    if rating is not None and not st.session_state.get("rating_logged"):
        with open("satisfaction_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "rating": rating + 1,
                "num_messages": len(st.session_state.messages),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }) + "\n")
        st.session_state.rating_logged = True
        st.success("Thank you for your feedback! / Merci pour votre retour !")