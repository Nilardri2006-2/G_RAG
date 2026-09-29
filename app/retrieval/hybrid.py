from app.retrieval.search import LegalRetriever
from app.retrieval.bm25_store import BM25Store

class HybridRetriever:
    def __init__(self):
        self.dense_retriever = LegalRetriever()
        self.bm25_retriever = BM25Store()

    def retrieve(
        self,
        query: str,
        dense_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 15
    ):
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=dense_k
        )
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=bm25_k
        )
        scores = {}
        documents = {}
        # dense retrieval results
        for rank, result in enumerate(dense_results, start=1):
            text = result.payload["text"]
            key = text
            scores[key] = scores.get(key, 0) + (
                1 / (60 + rank)
            )
            documents[key] = result.payload

        # BM25 retrieval results
        for rank, result in enumerate(bm25_results, start=1):
            text = result["chunk"]["text"]
            key = text
            scores[key] = scores.get(key, 0) + (
                1 / (60 + rank)
            )
            documents[key] = result["chunk"]

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for text, score in ranked[:final_k]:
            results.append({
                "text": text,
                "score": score,
                "metadata": documents[text]
            })
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "What is the Constitution (One Hundredth Amendment) Act, 2015?"
    results = retriever.retrieve(query)
    for i, result in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"RESULT {i}")
        print("RRF Score:", result["score"])
        print("Page:", result["metadata"]["page"])
        print("\n", result["text"][:700])