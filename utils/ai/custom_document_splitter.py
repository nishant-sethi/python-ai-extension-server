# custom class to create langchain DocumentSplitter at runtime

import datetime
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import time_taken


""" Split the documents into smaller chunks.

    Returns:
        docs_split: smaller chunks of the documents.
"""


class CustomDocumentSplitter:

    docs_split: list = []

    def __init__(self, text_splitter=RecursiveCharacterTextSplitter, chunk_size=1000, chunk_overlap=200, add_start_index=True, doc_loader=None):
        """ Initializes the CustomDocumentSplitter with the specified parameters.
            :param text_splitter: The text splitter to use (default: RecursiveCharacterTextSplitter).
            :param chunk_size: The size of the chunks (default: 1000).
            :param chunk_overlap: The overlap between chunks (default: 200).
            :param add_start_index: Whether to add the start index (default: True)
            :param doc_loader: The document loader to use (default: None).
        """
        self.text_splitter = text_splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

        # Initialize the document loader
        self.doc_loader = doc_loader

    def get_docs_split(self):
        """ Get the split documents.
        Returns:
            docs_split: The split documents.
        """
        return self.docs_split

    def split(self):
        """
        Splits the documents into smaller chunks.
        """
        docs = self.doc_loader.get_docs()
        if not docs:
            logging.error(f"""No document to split \n""")
            raise Exception(f"""No document to split""")
        try:
            text_splitter = self.text_splitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=self.add_start_index,
            )
            start_time = datetime.datetime.now()
            all_splits = text_splitter.split_documents(docs)
            self.docs_split = all_splits
            logging.info(f"""{len(all_splits)} splits created, Time Taken: {
                         time_taken(start_time)} \n""")
        except Exception as e:
            logging.error(f"""Failed to split documents: {
                          str(e)} \n""")
            raise Exception(f"""Failed to split documents: {
                str(e)}""")
