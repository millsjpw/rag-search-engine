from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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
        self.__map_documents(documents)
        formatted_list = []
        for i, doc in enumerate(documents):
            formatted = f"{doc['title']}: {doc['description']}"
            formatted_list.append(formatted)
            
        self.embeddings = self.model.encode(formatted_list, show_progress_bar=True)
        np.save(self.movie_path, self.embeddings)
        return self.embeddings
    
    def __map_documents(self, documents):
        for i, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.__map_documents(documents)
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