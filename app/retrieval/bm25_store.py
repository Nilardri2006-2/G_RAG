import json
import re
from rank_bm25 import BM25Okapi

CHUNKS_PATH = "data/processed/chunks.json"

def tokenize(text:str):
    return re.findall(r"\b\w+\b" , text.lower())

class BM25Store:
    def __init__(self):
        self.chunks = self.load_chunks()
        corpus=[
            tokenize(chunk["text"]) for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(corpus)

    def load_chunks(self):
        with open(CHUNKS_PATH , "r" , encoding="utf-8")as f:
            return json.load(f)

    def search(self, query:str , top_k: int = 10):
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1][:top_k]
        results = []
        for index in ranked_indices:
            results.append({
                "chunk" : self.chunks[index],
                "score" : float(scores[index]),
                "index" : int(index)
            })
        return results

if __name__ == "__main__":
    store = BM25Store()
    query = "What does article 25 protect?"
    results = store.search(query , top_k=5)
    for  i, result in enumerate(results):
        print(f"\nResult {i}:")
        print("Score:", result["score"])
        print("Page:", result["chunk"]["page"])
        print(result["chunk"]["text"][:500])

        