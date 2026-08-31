from pathlib import Path

from langchain_community.vectorstores import FAISS


INDEX_PATH = Path("data/index")


def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


def save_vector_store(vector_store):
    INDEX_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(str(INDEX_PATH))

    print("Vector store saved.")


def load_vector_store(embeddings):
    if not INDEX_PATH.exists():
        return None

    vector_store = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Vector store loaded.")

    return vector_store