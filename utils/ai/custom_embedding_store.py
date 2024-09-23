# custom class to create langchain EmbeddingStore at runtime
import datetime
import logging

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from utils.helpers import check_if_model_exists, time_taken


""" Store the split documents into a vector store.
    Returns:
        store: The vector store.
"""


class CustomEmbeddingStore:
    store = []

    def __init__(self, vector_store=Chroma, embedding=OllamaEmbeddings, model="nomic-embed-text", doc_splitter=None):
        """ Initializes the CustomEmbeddingStore with the specified parameters.
            :param vector_store: The vector store to use (default: Chroma).
            :param embedding: The embedding to use (default: OllamaEmbeddings).
            :param model: The model to use (default: "nomic-embed-text").
            :param doc_splitter: The document splitter to use (default: None).
        """
        self.vector_store = vector_store
        self.embedding = embedding
        self.model = model

        # Initialize the document splitter
        self.doc_splitter = doc_splitter

    def get_store(self):
        """ Get the stored vector store.

        Returns:
            store: The vector store.
        """
        return self.store

    def store_embeddings(self):
        """
        Stores the split documents into a vector store.
        """
        docs_split = self.doc_splitter.get_docs_split()
        if not docs_split:
            logging.error(f"""No document splits to store \n""")
            raise Exception(f"""No document splits to store""")
        try:
            start_time = datetime.datetime.now()
            logging.info(f"vectore store: {self.vector_store}")
            if not check_if_model_exists(self.model):
                logging.error(f"""Model Embedding {self.model} not found \n""")
                raise Exception(f"""Model Embedding {self.model} not found""")
            store = self.vector_store.from_documents(
                documents=docs_split, embedding=self.embedding(model=self.model))
            self.store = store
            logging.info(f"""\nStored documents, Time Taken: {
                         time_taken(start_time)} \n""")
        except Exception as e:
            logging.error(f"""Some error occurred while storing documents: {
                          str(e)} \n""")
            raise Exception(f"""Some error occurred while storing documents: {
                str(e)}""")
