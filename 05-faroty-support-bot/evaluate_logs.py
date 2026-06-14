import os, json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def judge(question, context, answer_text):
    p = f"""Evaluate this support answer.
Question: {question}
Context the assistant was given: {context}
Answer: {answer_text}
Score 1 to 5:
- faithfulness: is the answer supported by the context? 5 fully, 1 invented.
- relevance: does it address the question? 5 directly, 1 not at all.
Respond JSON only: {{"faithfulness": <int>, "relevance": <int>, "reason": "<short>"}}"""
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        contents=p)
    return json.loads(r.text)

faith, relev, n = 0, 0, 0
print("Low-scoring answers (these are your documentation gaps):\n")
with open("chat_logs.jsonl", encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        if e.get("action") != "answered":
            continue
        s = judge(e["question"], e.get("context", ""), e["answer"])
        faith += s["faithfulness"]; relev += s["relevance"]; n += 1
        if s["faithfulness"] <= 3:
            print(f"Q: {e['question']}\n   faith {s['faithfulness']} | {s['reason']}\n")

if n:
    print(f"\nEvaluated {n} answered interactions")
    print(f"Average faithfulness: {faith/n:.2f} / 5")
    print(f"Average relevance:    {relev/n:.2f} / 5")