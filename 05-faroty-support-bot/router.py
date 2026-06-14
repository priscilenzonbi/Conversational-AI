import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def classify_intent(message):
    prompt = f"""You are the routing brain of a support bot for Faroty, a fintech app
for groups, associations, tontines, and crowdfunding.

Classify the user's message into exactly one category:
- "general": a general question about how Faroty or ASSO+ works, that documentation could answer.
- "account_specific": about the user's OWN account, money, a transaction, a dispute, or a complaint. Needs a human with account access.
- "human_request": the user is explicitly asking to speak to a person.

Respond with JSON only: {{"category": "<category>", "reason": "<one short sentence>"}}

User message: {message}"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        contents=prompt,
    )
    return json.loads(resp.text)

# --- Test the brain in isolation ---
tests = [
    "How much commission does Faroty take on a solidarity campaign?",
    "Why hasn't my withdrawal of 50000 FCFA arrived in my account?",
    "I want to speak to someone please",
    "How do I create a tontine in ASSO+?",
    "Someone scammed me on a campaign, I want my money back",
]

for t in tests:
    result = classify_intent(t)
    print(f"{t}\n  -> {result['category']}: {result['reason']}\n")