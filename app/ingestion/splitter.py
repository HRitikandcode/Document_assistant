from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split LangChain Documents into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks