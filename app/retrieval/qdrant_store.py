from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_PATH = "storage/qdrant/"
COLLECTION_NAME = "legal_docs"
VECTOR_SIZE = 384

class QdrantStore:
    def __init__(self):
        Path(QDRANT_PATH).mkdir(
            parents = True,
            exist_ok = True
            )

        self.client = QdrantClient(
            path = QDRANT_PATH
        )

    def create_collection(self):
        collections = self.client.get_collections()
        existing_names = [
            collection.name
            for collection in collections.collections
        ]

        if COLLECTION_NAME not in existing_names:

            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

            print(
                f"Created collection: {COLLECTION_NAME}"
            )

        else:
            print(
                f"Collection already exists: {COLLECTION_NAME}"
            )
    def upsert(
        self,
        vectors,
        payloads,
    ):
        points = []

        for index, (vector, payload) in enumerate(
            zip(vectors, payloads)
        ):

            points.append(
                models.PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(
            f"Inserted {len(points)} vectors"
        )

    def search(
        self,
        query_vector,
        limit=5,
    ):

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        return results.points

