from typing import TypedDict

class LegalRAGState(TypedDict , total = False):
    query :str
    rewritten_query : str
    documents: list[dict]
    answer: str
    verification: dict
    retry_count: int
    max_retries: int
    verified: bool
    final_answer: str