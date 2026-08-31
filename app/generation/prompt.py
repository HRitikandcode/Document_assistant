from langchain_core.prompts import ChatPromptTemplate


def create_rag_prompt():

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an Engineering Knowledge Assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Do not use information that is not present in the context.
2. If the context does not contain enough information, say:
   "I don't have enough information in the provided documents."
3. Do not invent specifications, numbers, formulas, or technical facts.
4. Explain the answer clearly and technically.
5. When possible, refer to the source document and page number.

Context:

{context}
"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )

    return prompt