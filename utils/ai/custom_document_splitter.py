# custom class to create langchain DocumentSplitter at runtime

import datetime
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.ai.custom_document_loader import CustomDocumentLoader

logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s: line:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def log_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def time_taken(start_time):
    return f"""{(datetime.datetime.now() - start_time).total_seconds() * 1000:.3f}ms """


""" Split the documents into smaller chunks.

    Returns:
        docs_split: smaller chunks of the documents.
"""


class CustomDocumentSplitter:

    docs_split: list = []

    def __init__(self, text_splitter=RecursiveCharacterTextSplitter, chunk_size=1000,
                 chunk_overlap=200,
                 add_start_index=True):
        self.text_splitter = text_splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

    @classmethod
    def get_docs_split(cls):
        return cls.docs_split

    def split(self):
        """
        Splits the documents into smaller chunks.
        """
        docs = CustomDocumentLoader.get_docs()
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
            CustomDocumentSplitter.docs_split = all_splits
            logging.info(f"""{len(all_splits)} splits created, Time Taken: {
                         time_taken(start_time)} \n""")
        except Exception as e:
            logging.error(f"""Failed to split documents: {
                          str(e)} \n""")
            raise Exception(f"""Failed to split documents: {
                str(e)}""")
