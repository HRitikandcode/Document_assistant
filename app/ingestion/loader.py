from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(file_path: str):
    """
    Load a PDF and return LangChain Document objects.
    """
    
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    loader = PyMuPDFLoader(str(path))

    documents = loader.load()

    return documents