from app.retrieval.pipeline import LegalRetrievalPipeline
from app.generation.llm import CohereGenerator

class LegalRAG:

    def __init__(self):
        self.retriever = LegalRetrievalPipeline()
        self.generator = CohereGenerator()

    def answer(self, query: str):
        documents = self.retriever.retrieve(query)
        answer = self.generator.generate(
            query=query,
            documents=documents
        )
        return {
            "answer": answer,
            "sources": documents
        }


if __name__ == "__main__":
    rag = LegalRAG()
    while True:
        query = input(
            "\nAsk a legal question "
            "(type 'exit' to quit): "
        )
        if query.lower() == "exit":
            break
        result = rag.answer(query)

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(result["answer"])

        print("\n" + "=" * 80)
        print("RETRIEVED SOURCES")
        print("=" * 80)

        for i, source in enumerate(
            result["sources"],
            start=1
        ):

            print(
                f"\n{i}. "
                f"Page {source['metadata']['page']} "
                f"| Score {source['score']:.4f}"
            )