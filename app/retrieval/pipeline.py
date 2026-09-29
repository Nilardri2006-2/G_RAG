from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CohereReranker


class LegalRetrievalPipeline:
    def __init__(self):
        self.hybrid = HybridRetriever()
        self.reranker = CohereReranker()

    def retrieve(self, query: str):
        # stage 1
        candidates = self.hybrid.retrieve(
            query=query,
            dense_k=10,
            bm25_k=10,
            final_k=15
        )
        # stage 2
        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_n=5
        )

        return reranked


if __name__ == "__main__":

    pipeline = LegalRetrievalPipeline()
    query = "What is 100 Amendment Act??"
    results = pipeline.retrieve(query)
    for i, result in enumerate(results, start=1):
        print("\n" + "=" * 80)
        print(f"FINAL RESULT {i}")
        print(
            "Cohere Score:",
            round(result["score"], 4)
        )
        print(
            "Page:",
            result["metadata"]["page"]
        )
        print("\nText:")
        print(result["text"])