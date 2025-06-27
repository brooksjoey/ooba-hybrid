#!/usr/bin/env python3
import os, glob, logging
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(filename=os.path.expanduser("~/ooba-hybrid/logs/brain.log"), level=logging.INFO)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def ingest():
    docs_dir = os.path.expanduser("~/ooba-hybrid/memory/ingest/")
    persist_dir = os.path.expanduser("~/ooba-hybrid/memory/db")
    os.makedirs(docs_dir, exist_ok=True)

    loader = TextLoader
    docs = []
    for file in glob.glob(f"{docs_dir}/*.txt"):
        try:
            docs.extend(loader(file).load())
            logging.info(f"Loaded {file}")
        except Exception as e:
            logging.error(f"Failed to load {file}: {e}")

    db = Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )
    db.persist()
    logging.info("[✓] Ingest complete.")

if __name__ == "__main__":
    ingest()

# Save this script
with open(os.path.expanduser("~/ooba-hybrid/scripts/ingest_context.py"), "w") as f:
    f.write(__file__)
os.chmod(os.path.expanduser("~/ooba-hybrid/scripts/ingest_context.py"), 0o755)
