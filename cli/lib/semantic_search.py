from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
import json
from .search_utils import CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.movie_path = f"{CACHE_DIR}/movie_embeddings.npy"
        
    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("Input text cannot be empty")
        return self.model.encode([text])[0]
        
    def build_embeddings(self, documents):
        if not documents:
            raise ValueError("Documents list cannot be empty")
        self.documents = documents
        self.map_documents(documents)
        formatted_list = []
        for i, doc in enumerate(documents):
            formatted = f"{doc['title']}: {doc['description']}"
            formatted_list.append(formatted)
            
        self.embeddings = self.model.encode(formatted_list, show_progress_bar=True)
        np.save(self.movie_path, self.embeddings)
        return self.embeddings
    
    def map_documents(self, documents):
        for i, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.map_documents(documents)
        if os.path.exists(self.movie_path):
            self.embeddings = np.load(self.movie_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("Embeddings and documents must be loaded before searching")

        query_embedding = self.generate_embedding(query)
        similarities = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        # create a list of (document, similarity) tuples
        doc_similarities = list(zip(self.documents, similarities))
        # sort by similarity in descending order
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        # return top results up to the limit, containing score, title, description
        return [{"score": sim, "title": doc["title"], "description": doc["description"]} for doc, sim in doc_similarities[:limit]]
    
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.map_documents(documents)
        all_chunks = []
        chunk_metadata = []
        
        for i, doc in enumerate(documents):
            if "description" not in doc:
                continue
            semantic_chunks = semantic_chunking(doc["description"], chunk_size=4, overlap=1)
            all_chunks.extend(semantic_chunks)
            chunk_metadata.extend([{"movie_idx": i, "chunk_idx": j, "total_chunks": len(semantic_chunks)} for j in range(len(semantic_chunks))])
        
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(f"{CACHE_DIR}/chunk_embeddings.npy", self.chunk_embeddings)
        with open(f"{CACHE_DIR}/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
            
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.map_documents(documents)
        if os.path.exists(f"{CACHE_DIR}/chunk_embeddings.npy") and os.path.exists(f"{CACHE_DIR}/chunk_metadata.json"):
            self.chunk_embeddings = np.load(f"{CACHE_DIR}/chunk_embeddings.npy")
            with open(f"{CACHE_DIR}/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            if len(self.chunk_embeddings) == sum(len(semantic_chunking(doc["description"], chunk_size=4, overlap=1)) for doc in documents if "description" in doc):
                return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.documents is None:
            raise ValueError("Chunk embeddings and documents must be loaded before searching")

        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, emb in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, emb)
            chunk_scores.append({
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score
            })
            
        movie_scores = {}
        for chunk in chunk_scores:
            movie_idx = chunk["movie_idx"]
            if movie_idx not in movie_scores or chunk["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = chunk
        sorted_movies = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)[:limit]
        return [{
            "id": movie_idx, 
            "title": self.documents[movie_idx]["title"],
            "document": self.documents[movie_idx]["description"][:100],
            "score": round(movie_scores[movie_idx]["score"], 4),
            "metadata": movie_scores[movie_idx] or {}
        } for movie_idx in [movie["movie_idx"] for movie in sorted_movies]]

        
        

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
    
def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    
def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    results = ss.search(query, limit)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} ({result['score']:.4f})\n\t{result['description']}\n")
        
def chunk_command(text: str, chunk_size: int, overlap: int):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
        
def fixed_size_chunking(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks

def semantic_chunking(text: str, chunk_size: int, overlap: int):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and sentences[0][len(sentences[0]) - 1] not in ".!?":
        return [sentences[0]]
    
    i = 0
    chunks = []
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        combined = " ".join(chunk_sentences).strip()
        if combined:
            chunks.append(combined)

        i += chunk_size - overlap

    return chunks

def semantic_chunk_command(text: str, chunk_size: int, overlap: int):
    chunks = semantic_chunking(text, chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
        
def embed_chunks_command():
    css = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = css.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
    
def search_chunked_command(query: str, limit: int):
    css = ChunkedSemanticSearch()
    documents = load_movies()
    css.load_or_create_chunk_embeddings(documents)
    results = css.search_chunks(query, limit)
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")