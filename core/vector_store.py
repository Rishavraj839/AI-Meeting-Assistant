from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

CHROMA_DIR="VECTOR_DB"
COLLECTION_NAME="meeting_transcripts"
EMBEDDING_MODEL="all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,model_kwargs={"device": "cpu"} )

def build_vector_store(transcript: str) -> Chroma:
    """Build a Chroma vector store from a meeting transcript."""
    print("Building vector store...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    
    docs = [Document(page_content=chunk, 
                          metadata={'chunk_index': i}) for i, chunk in enumerate(chunks)]
    embeddings = get_embeddings()
    
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        collection_name=COLLECTION_NAME, 
        persist_directory=CHROMA_DIR)
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing Chroma vector store."""
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME, 
        embedding_function=embeddings, 
        persist_directory=CHROMA_DIR)
    return vector_store

def get_retriever(vector_store: Chroma , k: int = 4):
    """Get a retriever from the Chroma vector store."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
        )
    