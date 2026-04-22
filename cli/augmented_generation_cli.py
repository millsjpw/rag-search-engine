import argparse

from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize top search results for a query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top search results to summarize"
    )
    
    citations_parser = subparsers.add_parser(
        "citations", help="Answer query with citations from search results"
    )
    citations_parser.add_argument("query", type=str, help="Search query for answer with citations")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top search results to use for answering"
    )
    
    question_parser = subparsers.add_parser(
        "question", help="Answer a question based on search results without citations"
    )
    question_parser.add_argument("question", type=str, help="Question to answer based on search results")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top search results to use for answering the question"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            docs, answer = rag_command(query)
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("RAG Response:")
            print(answer)
        case "summarize":
            docs, summary = summarize_command(args.query, args.limit)
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("LLM Summary:")
            print(summary)
        case "citations":
            docs, answer = citations_command(args.query, args.limit)
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("LLM Answer:")
            print(answer)
        case "question":
            docs, answer = question_command(args.question, args.limit)
            print("Search Results:")
            for doc in docs:
                print(f"- {doc['title']}")
            print()
            print("Answer:")
            print(answer)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()