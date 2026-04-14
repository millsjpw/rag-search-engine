import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

            
    def load(self) -> None:
        # raising errors if the files do not exist
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path) or not os.path.exists(self.term_freq_path) or not os.path.exists(self.doc_lengths_path):
            raise FileNotFoundError("Index, docmap, term frequencies, or document lengths file does not exist.")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_freq_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
            
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
            
    def get_tf(self, doc_id, term):
        # raise exception if term is more than one word
        if len(tokenize_text(term)) != 1:
            raise ValueError("Term must be a single word.")

        term = tokenize_text(term)[0]  # assuming term is a single word
        return self.term_frequencies.get(doc_id, {}).get(term, 0)
    
    def get_idf(self, term):
        # raise exception if term is more than one word
        if len(tokenize_text(term)) != 1:
            raise ValueError("Term must be a single word.")

        term = tokenize_text(term)[0]  # assuming term is a single word
        doc_count = len(self.index.get(term, []))
        if doc_count == 0:
            return 0.0
        return math.log((len(self.docmap) + 1) / (doc_count + 1))  # added 1 to avoid division by zero
    
    def get_bm25_idf(self, term):
        # raise exception if term is more than one word
        if len(tokenize_text(term)) != 1:
            raise ValueError("Term must be a single word.")

        term = tokenize_text(term)[0]  # assuming term is a single word
        doc_count = len(self.index.get(term, []))
        if doc_count == 0:
            return 0.0
        return math.log((len(self.docmap) - doc_count + 0.5) / (doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        # raise exception if term is more than one word
        if len(tokenize_text(term)) != 1:
            raise ValueError("Term must be a single word.")

        term = tokenize_text(term)[0]  # assuming term is a single word
        tf = self.term_frequencies.get(doc_id, {}).get(term, 0)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length)) if tf > 0 else 0.0
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query: str, limit: int):
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = sum(self.bm25(doc_id, token) for token in tokens)
            if score > 0:
                scores[doc_id] = score
        # return the top limit documents and their scores
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"id": doc_id, "title": self.docmap[doc_id]["title"], "score": score} for doc_id, score in ranked_docs]


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    # if index doesn't exist print an error and exit
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return []

    results = []
    for term in tokenize_text(query):
        for doc_id in idx.get_documents(term):
            results.append({"id": doc_id, "title": idx.docmap[doc_id]["title"], "score": idx.bm25(doc_id, term)})
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return results

def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Document ID: {doc_id}, Term: {term}, TF: {idx.get_tf(doc_id, term)}")
    
def idf_command(term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Term: {term}, IDF: {idx.get_idf(term):.2f}")
    
def tfidf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return

    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)
    tf_idf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
    
def bm25_idf_command(term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"BM25 IDF score of '{term}': {idx.get_bm25_idf(term):.2f}")
    
def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"BM25 TF score of '{term}' in document '{doc_id}': {idx.get_bm25_tf(doc_id, term, k1=k1, b=b):.2f}")
    
def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return

    results = idx.bm25_search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")

    

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
