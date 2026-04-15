#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_command, semantic_chunk_command, embed_chunks_command, search_chunked_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")
    
    embed_parser = subparsers.add_parser("embed_text", help="Embed a given text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings for the documents")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a given query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser("search", help="Search for documents matching a query")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", nargs='?', default=DEFAULT_SEARCH_LIMIT,type=int, help="Number of top results to return")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a given text into smaller pieces semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=200, help="Size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for chunked documents")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for chunked documents matching a query")
    search_chunked_parser.add_argument("query", type=str, help="Query text to search for")
    search_chunked_parser.add_argument("--limit", nargs='?', default=DEFAULT_SEARCH_LIMIT,type=int, help="Number of top results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()