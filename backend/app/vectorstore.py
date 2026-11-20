from functools import lru_cache
from langchain_chroma import Chroma
from app.config import settings
from app.embeddings import get_embeddings

COLLECTION_NAME="documents_v1"

@lru_cache(maxsize=1)
def get_vector_store()-> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        # Make distance explicit 0=identical, bigger=less similar
        collection_metadata={"hnsw:space": "cosine"},
    )

def delete_document_chunks(doc_id: str)-> None:
    """
    Remove every chunk that belongs to one document
    """
    get_vector_store()._collection.delete(where={"doc_id":doc_id})

def count_chunks()-> int:
    return get_vector_store()._collection.count()
