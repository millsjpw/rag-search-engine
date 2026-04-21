import os

from .keyword_search import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch
from .gemini import spellcheck_query, rewrite_query, expand_query, rate_against_query, batch_rerank_against_query
from time import sleep
import json

from sentence_transformers import CrossEncoder


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        combined = combine_search_results_rrf(bm25_results, semantic_results, k)
        return combined[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(bm25_rank: int, semantic_rank: int, k: int = DEFAULT_K) -> float:
    rrf_bm25 = 1 / (k + bm25_rank) if bm25_rank > 0 else 0
    rrf_semantic = 1 / (k + semantic_rank) if semantic_rank > 0 else 0
    return rrf_bm25 + rrf_semantic

def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

def combine_search_results_rrf(bm25_results: list[dict], semantic_results: list[dict], k: int = DEFAULT_K) -> list[dict]:
    bm25_results = sorted(bm25_results, key=lambda x: x["score"], reverse=True)
    semantic_results = sorted(semantic_results, key=lambda x: x["score"], reverse=True)
    combined_scores = {}

    for i, result in enumerate(bm25_results):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": i + 1,
                "semantic_rank": 0,
            }

    for i, result in enumerate(semantic_results):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": 0,
                "semantic_rank": i + 1,
            }
        else:
            combined_scores[doc_id]["semantic_rank"] = i + 1

    rrf_results = []
    for doc_id, data in combined_scores.items():
        score_value = rrf_score(data["bm25_rank"], data["semantic_rank"], k)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return sorted(rrf_results, key=lambda x: x["score"], reverse=True)



def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }

def rrf_search_command(
    query: str, k: int = DEFAULT_K, limit: int = DEFAULT_SEARCH_LIMIT, enhance: str = None, rerank_method: str = None
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    if enhance == "spell":
        enhanced_query = spellcheck_query(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'")
        query = enhanced_query
        
    elif enhance == "rewrite":
        enhanced_query = rewrite_query(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'")
        query = enhanced_query
        
    elif enhance == "expand":
        enhanced_query = expand_query(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'")
        query = enhanced_query
        
    if rerank_method:
        search_limit = limit * 5
    else:
        search_limit = limit
    results = searcher.rrf_search(query, k, search_limit)

    if rerank_method:
        if rerank_method == "individual":
            for doc in results:
                doc["re-rank"] = rate_against_query(query, doc.get("title", ""), doc.get("document", ""))
                sleep(3)
            results = sorted(results, key=lambda x: x.get("re-rank", x["score"]), reverse=True)[:limit]
        elif rerank_method == "batch":
            doc_ids = json.loads(batch_rerank_against_query(query, results))
            # Assuming the response is a JSON list of doc_ids in the order of relevance
            # sort results based on the order of doc_ids returned by the batch reranking
            # and assign "re-rank" as the position, i.e. 1, 2, 3, etc.
            doc_id_to_rank = {doc_id: rank for rank, doc_id in enumerate(doc_ids, start=1)}
            for doc in results:                
                doc_id = doc["id"]
                doc["re-rank"] = doc_id_to_rank.get(doc_id, len(results) + 1)  # If not found, assign a rank worse than all results
            # sort results based on "re-rank"
            results = sorted(results, key=lambda x: x.get("re-rank", x["score"]))[:limit]
        elif rerank_method == "cross_encoder":
            pairs = [(query, f"{doc.get('title', '')} - {doc.get('document', '')}") for doc in results]
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L2-v2')
            scores = cross_encoder.predict(pairs)
            for i, doc in enumerate(results):
                doc["re-rank"] = scores[i]
            results = sorted(results, key=lambda x: x.get("re-rank", x["score"]), reverse=True)[:limit]
            

    return {
        "original_query": original_query,
        "query": query,
        "k": k,
        "results": results,
    }
