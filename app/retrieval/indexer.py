import json

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.qdrant_store import QdrantStore


CHUNKS_PATH = "data/processed/chunks.json"


def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("loading the chunks...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("loading embedding model....")

    embedding_model = EmbeddingModel()

    print("generating these embeddings...")

    vectors = embedding_model.embed_documents(
        texts
    )

    print(
        f"Generated embeddings: {vectors.shape}"
    )

    payloads = []

    for chunk in chunks:

        payloads.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "document_type": chunk["document_type"],
        })

    store = QdrantStore()

    store.create_collection()

    store.upsert(
        vectors=vectors,
        payloads=payloads,
    )

    print("\nIndexing complete.")


if __name__ == "__main__":
    main()