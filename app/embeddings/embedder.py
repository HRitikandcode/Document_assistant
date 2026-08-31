from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def create_embedder():

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return embeddings