import json
import re
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

class LegalVerifier:
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

    def verify(
        self,
        query: str,
        answer: str,
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
Page:{metadata["page"]}
Source:{metadata["source"]}
Content:{document["text"]}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a strict verifier for an Indian legal RAG system.
USER QUESTION: {query}
GENERATED ANSWER: {answer}
RETRIEVED LEGAL CONTEXT: {context}

Determine whether the answer is adequately supported by the retrieved
legal context.

Check:

1. Whether the major legal claims are supported.
2. Whether the answer actually answers the question.
3. Whether the cited pages correspond to the retrieved evidence.
4. Whether the answer contains unsupported claims or hallucinations.
5. Whether another retrieval attempt could improve the answer.

Return ONLY valid JSON in this exact format:

{{
    "supported": true,
    "reason": "short explanation",
    "missing_information": [],
    "rewritten_query": ""
}}

If the answer is NOT adequately supported:

- supported must be false
- explain what evidence is missing
- provide a better search query in rewritten_query

If the answer IS adequately supported:

- supported must be true
- rewritten_query should be an empty string
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        raw_text = self.extract_text(response)

        return self.parse_json(raw_text)

    def parse_json(self, text):

        text = text.strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            pass

        # Handle ```json ... ``` if the model adds markdown
        match = re.search(
            r"```json\s*(.*?)\s*```",
            text,
            re.DOTALL
        )

        if match:

            return json.loads(
                match.group(1)
            )

        raise ValueError(
            f"Verifier did not return valid JSON:\n{text}"
        )