from app.graph.legal_graph import build_graph


def main():

    graph = build_graph()
    print("=" * 80)
    print("SELF-HEALING LEGAL RAG")
    print("=" * 80)
    while True:
        query = input(
            "\nLegal question "
            "(type 'exit' to quit): "
        )
        if query.lower() == "exit":
            break
        initial_state = {
            "query": query,
            "retry_count": 0,
            "max_retries": 2,
            "verified": False
        }
        result = graph.invoke(
            initial_state
        )
        print("\n" + "=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)
        print(
            result.get(
                "answer",
                "No answer generated."
            )
        )
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        print(
            result.get(
                "verification"
            )
        )
        print(
            "\nRetries:",
            result.get("retry_count", 0)
        )

if __name__ == "__main__":
    main()