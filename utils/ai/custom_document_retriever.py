
import logging

from utils.custom_exceptions import DocumentRetrieverException, GetDocumentRetrieverException


class CustomDocumentRetriever:
    """Custom class to create a Langchain DocumentRetriever at runtime."""

    def __init__(self, search_type="similarity", search_args={"k": 5}, embedding_store=None):
        """
        Initializes the CustomDocumentRetriever with specified search parameters.

        :param search_type: The type of search to use (default: "similarity").
        :param search_args: Arguments for the search (default: {"k": 5}).
        :param embedding_store: The embedding store to use (default: None).
        """
        self.search_type = search_type
        self.search_args = search_args
        self.retriever = None  # Instance variable to store retriever

        # Initialize the embedding store
        self.embedding_store = embedding_store

    def get_retriever(self):
        """
        Retrieves the retriever object.

        :return: The retriever object.
        :raises Exception: If the retriever is not set up.
        """
        if not self.retriever:
            logging.error(f""" Retriever not set up \n""")
            raise GetDocumentRetrieverException(f"""Retriever not set up.""")
        return self.retriever

    def retrieve(self):
        """
        Sets up a retriever from the vector store.
        """
        try:
            vector_store = self.embedding_store.get_store()
            logging.info(f"Retrieved vector store ")
        except Exception as e:
            logging.error(f"""No vector store to retrieve from: {str(e)} \n""")
            raise DocumentRetrieverException(f"""No vector store to retrieve from: {str(e)}""")

        try:
            logging.info(f"Initializing retriever ")
            retriever = vector_store.as_retriever(
                search_type=self.search_type, search_args=self.search_args)
            self.retriever = retriever
            logging.info(f"Initialized retriever \n")
        except Exception as e:
            logging.error(f"Failed to initialize retriever: {str(e)} \n")
            raise DocumentRetrieverException(f"""Failed to initialize retriever: {str(e)}""")
