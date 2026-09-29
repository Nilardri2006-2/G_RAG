import os
import cohere
from dotenv import load_dotenv

load_dotenv()

class CohereReranker:
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not found in environment"
            )
        self.client = cohere.ClientV2(
            api_key=api_key
        )
        self.model = "rerank-v4.0-pro"

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int = 5
    ):
        texts = [
            document["text"]
            for document in documents
        ]
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=top_n
        )

        results = []
        for result in response.results:
            original_document = documents[result.index]
            results.append({
                "text": original_document["text"],
                "score": result.relevance_score,
                "metadata": original_document["metadata"]
            })

        return results