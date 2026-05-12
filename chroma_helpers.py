"""Compatibility shim: vector store is implemented in vector_store.py."""

from vector_store import (  # noqa: F401
    STREAMLIT_REGISTRY_COLLECTION,
    create_chroma_client,
    create_vector_client,
    delete_named_collection,
    get_or_create_vector_collection,
    load_streamlit_file_registry,
    save_streamlit_file_registry,
)
