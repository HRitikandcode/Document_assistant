from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.embeddings.embedder import create_embedder
from app.retrieval.vector_store import create_vector_store, save_vector_store


PDF_PATH = "data/document/doc_1.pdf"


def build_index():

    print("Loading PDF...")

    documents = load_pdf(PDF_PATH)

    print(f"Loaded {len(documents)} pages.")

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embedding model...")

    embeddings = create_embedder()

    print("Creating vector store...")

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    save_vector_store(vector_store)

    print("Indexing complete.")


if __name__ == "__main__":
    build_index()