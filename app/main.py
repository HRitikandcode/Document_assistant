from app.embeddings.embedder import create_embedder
from app.retrieval.vector_store import load_vector_store
from app.retrieval.retriever import create_retriever

from app.generation.llm import create_llm
from app.generation.prompt import create_rag_prompt
from app.generation.rag_chain import create_rag_chain


def main():

    # -----------------------------------------
    # 1. Create embedding model
    # -----------------------------------------

    embeddings = create_embedder()

    # -----------------------------------------
    # 2. Load FAISS index
    # -----------------------------------------

    vector_store = load_vector_store(
        embeddings
    )

    if vector_store is None:

        print("Vector store not found.")

        print(
            "Run index_documents.py first."
        )

        return

    # -----------------------------------------
    # 3. Create LangChain retriever
    # -----------------------------------------

    retriever = create_retriever(
        vector_store
    )

    # -----------------------------------------
    # 4. Create LLM
    # -----------------------------------------

    llm = create_llm()

    # -----------------------------------------
    # 5. Create prompt
    # -----------------------------------------

    prompt = create_rag_prompt()

    # -----------------------------------------
    # 6. Create RAG chain
    # -----------------------------------------

    rag_chain = create_rag_chain(
        retriever,
        prompt,
        llm
    )

    # -----------------------------------------
    # 7. Ask questions
    # -----------------------------------------

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        answer = rag_chain.invoke(
            question
        )

        print("\n===== ANSWER =====")
        print(answer)


if __name__ == "__main__":
    main()