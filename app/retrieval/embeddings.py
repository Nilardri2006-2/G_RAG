from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingModel:
    def __init__(self):
        print(f" Loadin the embedding model : {MODEL_NAME}")
        self.model  = SentenceTransformer(MODEL_NAME)

    def embed_documents(self , texts: list[str]):
        """
        its gonna generate embeddings for the doc chunks
        """
        return self.model.encode(
            texts,
            batch_size = 32,
            show_progress_bar = True,
            # normalize_embbeddings = True,
        )
    def embed_query(self, query:str):
        """
        it will generate embeddings for the user query
        """
        return self.model.encode(
            query,
            # normalize_embeddings = True,
        )