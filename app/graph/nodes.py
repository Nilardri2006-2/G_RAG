from app.retrieval.pipeline import LegalRetrievalPipeline
from app.generation.llm import CohereGenerator
from app.generation.verifier import LegalVerifier

retriever = LegalRetrievalPipeline()
generator = CohereGenerator()
verifier = LegalVerifier()

def retrieve_node(state):
    query = state.get(
        "rewritten_query",
        state["query"]
    )
    print("\n[RETRIEVAL]")
    print("query:", query)

    documents = retriever.retrieve(
        query
    )

    return {
        "documents": documents
    }


def generate_node(state):
    query = state.get(
        "rewritten_query",
        state["query"]
    )
    print("\n[generation]")
    answer = generator.generate(
        query=query,
        documents=state["documents"]
    )
    return {
        "answer": answer
    }

def verify_node(state):
    query = state.get(
        "rewritten_query",
        state["query"]
    )
    print("\n[verification]")
    verification = verifier.verify(
        query=query,
        answer=state["answer"],
        documents=state["documents"]
    )
    print(
        "Supported:",
        verification.get("supported")
    )
    print(
        "Reason:",
        verification.get("reason")
    )
    return {
        "verification": verification,
        "verified": verification.get(
            "supported",
            False
        )
    }

def rewrite_node(state):
    verification = state["verification"]
    rewritten_query = verification.get(
        "rewritten_query",
        ""
    )
    retry_count = state.get(
        "retry_count",
        0
    )
    print("\n[SELF-HEALING]")
    print("Rewritten query:", rewritten_query)
    return {
        "rewritten_query": rewritten_query,
        "retry_count": retry_count + 1
    }