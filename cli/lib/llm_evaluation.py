import json
import os
from time import sleep

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"

def llm_evaluate_relevance(query: str, formatted_results: list[dict]):
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {json.dumps(formatted_results, indent=2)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""
    
    response = client.models.generate_content(model=model, contents=prompt)
    scores_text = (response.text or "").strip()
    
    scores = json.loads(scores_text)
    # match each score to the corresponding document
    scored_results = []
    for result, score in zip(formatted_results, scores):
        scored_results.append({**result, "relevance_score": score})
    return scored_results