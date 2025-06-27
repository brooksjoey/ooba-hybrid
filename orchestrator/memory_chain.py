import os
import logging
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
# Placeholder: Optional advanced backends
# from langchain_community.vectorstores import Qdrant, Weaviate

logging.basicConfig(
    filename=os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

class MemoryAdapter:
    def __init__(self, backend: str = "chroma"):
        self.persist_directory = os.path.expanduser("~/ooba-hybrid/memory/db")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.embedding_function: Embeddings = OpenAIEmbeddings()
        self.backend = backend.lower()
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> VectorStore:
        try:
            if self.backend == "chroma":
                return Chroma(
                    collection_name="ooba-memory",
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory,
                )
            # elif self.backend == "qdrant":
            #     return Qdrant.from_documents(...)
            # elif self.backend == "weaviate":
            #     return Weaviate.from_documents(...)
            else:
                raise ValueError(f"Unsupported memory backend: {self.backend}")
        except Exception as e:
            logging.error(f"Failed to initialize vectorstore backend '{self.backend}': {e}")
            raise

    def similarity_search(self, query: str, k: int = 3):
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            return []

    def add_documents(self, docs):
        try:
            self.vectorstore.add_documents(docs)
        except Exception as e:
            logging.error(f"Failed to add documents to memory: {e}")

def get_memory() -> VectorStore:
    adapter = MemoryAdapter(backend="chroma")  # Can be swapped to "qdrant", "weaviate", etc.
    return adapter
