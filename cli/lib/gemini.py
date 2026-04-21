import os
from dotenv import load_dotenv
from google import genai

def ask_gemini(prompt):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=[
            prompt
        ])
    
    return response.text.strip()

def spellcheck_query(query):
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    
    return ask_gemini(prompt)

def rewrite_query(query):
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """
    
    return ask_gemini(prompt)

def expand_query(query):
    prompt = f"""Expand the user-provided movie search query below by adding relevant keywords to improve search results.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the expanded query concise (under 15 words total)
    - It should be a Google-style search query, specific enough to yield relevant results

    Examples:
    - "that bear movie where leo gets attacked" -> "that bear movie where leo gets attacked The Revenant Leonardo DiCaprio"
    - "movie about bear in london with marmalade" -> "movie about bear in london with marmalade Paddington"
    - "scary movie with bear from few years ago" -> "scary movie with bear from few years ago bear horror movie 2015-2020"
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"
    - "math movie" -> "math movie Good Will Hunting I.Q. A Beautiful Mind"

    If you cannot improve the query, output the original unchanged.
    Output only the expanded query text, nothing else.

    User query: "{query}"
    """
    
    return ask_gemini(prompt)

def rate_against_query(query, title, document):
    prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {title} - {document}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""
    
    response = ask_gemini(prompt)
    try:
        score = float(response)
        return max(0, min(10, score))  # Ensure score is between 0 and 10
    except ValueError:
        return 0  # If response isn't a valid number, return 0 as default
    
def batch_rerank_against_query(query, results):
    doc_list_str = "\n".join([f"{doc['id']}: {doc['title']} - {doc['document']}" for doc in results])
    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""
    
    return ask_gemini(prompt)