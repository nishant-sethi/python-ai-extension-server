# custom class to create langchain EmbeddingStore at runtime
import datetime
import logging

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from utils.ai.custom_document_splitter import CustomDocumentSplitter

logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s: line:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def time_taken(start_time):
    return f"""{(datetime.datetime.now() - start_time).total_seconds() * 1000:.3f}ms """


""" Store the split documents into a vector store.
    Returns:
        store: The vector store.
"""


class CustomEmbeddingStore:
    store = []

    def __init__(self, vector_store=Chroma, embedding=OllamaEmbeddings, model="nomic-embed-text"):
        self.vector_store = vector_store
        self.embedding = embedding
        self.model = model

    @classmethod
    def get_store(cls):
        return cls.store

    def store(self):
        """
        Stores the split documents into a vector store.
        """
        docs_split = CustomDocumentSplitter.get_docs_split()
        if not docs_split:
            logging.error(f"""No document splits to store \n""")
            raise Exception(f"""No document splits to store""")
        try:
            start_time = datetime.datetime.now()
            store = self.vector_store.from_documents(
                documents=docs_split, embedding=self.embedding(model=self.model))
            CustomEmbeddingStore.store = store
            logging.info(f"""\nStored documents, Time Taken: {
                         time_taken(start_time)} \n""")
        except Exception as e:
            logging.error(f"""Some error occurred while storing documents: {
                          str(e)} \n""")
            raise Exception(f"""Some error occurred while storing documents: {
                str(e)}""")
