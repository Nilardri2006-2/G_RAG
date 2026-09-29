import os
import cohere
from dotenv import load_dotenv

load_dotenv()


class CohereGenerator:

    def __init__(self):

        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not found"
            )

        self.client = cohere.ClientV2(
            api_key=api_key
        )

        self.model = "command-a-plus-05-2026"

    def extract_text(self, response):

        for item in response.message.content:

            if item.type == "text":
                return item.text

        raise RuntimeError(
            "No text content returned by Cohere"
        )

    def generate(
        self,
        query: str,
        documents: list[dict]
    ):

        context_parts = []

        for i, document in enumerate(
            documents,
            start=1
        ):

            metadata = document["metadata"]

            context_parts.append(
                f"""
SOURCE {i}

Document:
{metadata["source"]}

Document Type:
{metadata["document_type"]}

Page:
{metadata["page"]}

Content:
{document["text"]}
"""
            )

        context = "\n".join(context_parts)

        system_prompt = """
You are a legal research assistant specializing in Indian law.

Answer questions ONLY using the supplied legal context.

Rules:

1. Do not invent legal provisions.
2. Do not use information outside the supplied context.
3. Cite the relevant document and page number.
4. If the context does not contain enough information, say:
   "The retrieved sources do not provide enough information to answer this question."
5. Distinguish clearly between what the source states and your explanation.
6. Keep the answer precise and legally grounded.
"""

        user_prompt = f"""
LEGAL CONTEXT:

{context}

QUESTION:

{query}

Provide a concise answer with citations.
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0
        )

        return self.extract_text(response)