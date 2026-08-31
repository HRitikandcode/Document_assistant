from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_documents(documents):

    formatted_documents = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        text = document.page_content

        formatted_documents.append(
            f"""
Source: {source}
Page: {page}

{text}
"""
        )

    return "\n\n".join(formatted_documents)


def create_rag_chain(retriever, prompt, llm):

    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain