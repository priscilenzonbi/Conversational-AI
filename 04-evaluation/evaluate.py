import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- The knowledge base we are testing against ---
documents = [
    "Transfers between PayFlow accounts are free. Transfers to external banks cost 1.5% of the amount, capped at 5 dollars.",
    "A PayFlow account can be opened by anyone aged 18 or older with a valid national ID. There is no minimum opening balance.",
    "If your card is lost or stolen, freeze it in the app under Cards then Freeze. A replacement card costs 3 dollars and arrives in 5 working days.",
    "PayFlow does not offer loans or credit. We provide current accounts, savings pots, and card services only.",
    "Customer support is available 24/7 through in-app chat. Phone support runs 8am to 8pm, Monday to Friday.",
]

def embed_texts(texts):
    r = client.models.embed_content(model="gemini-embedding-001", contents=texts)
    return [np.array(e.values) for e in r.embeddings]

doc_vectors = embed_texts(documents)

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(question, top_k=2):
    q = client.models.embed_content(model="gemini-embedding-001", contents=question)
    qv = np.array(q.embeddings[0].values)
    scores = [similarity(qv, v) for v in doc_vectors]
    order = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in order]

def answer(question):
    context = "\n".join(retrieve(question))
    prompt = f"""Answer using ONLY the information below.
If the answer is not in the information, say you do not have that information.

Information:
{context}

Question: {question}"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="You are a concise PayFlow support assistant."
        ),
        contents=prompt,
    )
    return resp.text, context

# --- THE JUDGE: a model scoring the answer ---
def judge(question, context, answer_text):
    judge_prompt = f"""You are evaluating an AI assistant's answer.

Question: {question}
Context the assistant was given: {context}
Assistant's answer: {answer_text}

Score two things from 1 to 5:
- faithfulness: is the answer fully supported by the context? 5 means fully grounded, 1 means invented.
- relevance: does the answer address the question? 5 means directly, 1 means not at all.

Respond with JSON only: {{"faithfulness": <int>, "relevance": <int>, "reason": "<one short sentence>"}}"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        contents=judge_prompt,
    )
    return json.loads(resp.text)

# --- The test set: questions plus what we expect ---
test_set = [
    "How much does sending money to another bank cost?",
    "Can a 16 year old open a PayFlow account?",
    "How do I freeze a stolen card?",
    "Does PayFlow offer home loans?",
    "What is the capital of France?",
]

print(f"{'Question':<45} {'Faith':<6} {'Relev':<6} Reason")
print("-" * 100)
faith_total, relev_total = 0, 0
for q in test_set:
    ans, ctx = answer(q)
    scores = judge(q, ctx, ans)
    faith_total += scores["faithfulness"]
    relev_total += scores["relevance"]
    print(f"{q[:43]:<45} {scores['faithfulness']:<6} {scores['relevance']:<6} {scores['reason']}")

n = len(test_set)
print("-" * 100)
print(f"Average faithfulness: {faith_total/n:.2f} / 5")
print(f"Average relevance:    {relev_total/n:.2f} / 5")