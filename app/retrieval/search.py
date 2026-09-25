from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.qdrant_store import QdrantStore

class LegalRetriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.store = QdrantStore()

    def retrieve(
            self,
            query:str,
            top_k: int=5,
    ):
        query_vector = (
            self.embedding_model.embed_query(query)
        )

        results = self.store.search(
            query_vector = query_vector,
            limit = top_k,
        )
        return results

def main():
        retriever = LegalRetriever()

        while True:
            query = input(
                "\n Ask a legal question (type 'exit' to quit): "
            )

            if query.lower() == "exit":
                break
            results = retriever.retrieve(
                query,
                top_k=5,
            )
            print("\n" + "=" *70)
            print("retrieved results: ")
            print("=" * 70)

            for i, result in enumerate(
                results,
                start=1,
            ):
                print(f"\n RESULT {i}")
                print(f"\n SCORE {result.score:.4f}")
                print(
                    f"Source: ",
                    f"{result.payload.get('source')}")

                print(
                    f"Page: "
                    f"{result.payload.get('page')}"
                )
                print("\nText:")
                print(result.payload.get("text"))
                print("-" * 70)

if __name__ == "__main__":
     main()