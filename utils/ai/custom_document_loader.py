# custom class to create langchain DocumentLoader at runtime

import datetime
import logging
import bs4
from langchain_community.document_loaders import WebBaseLoader

logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s: line:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def time_taken(start_time):
    return f"""{(datetime.datetime.now() - start_time).total_seconds() * 1000:.3f}ms """


""" Load the documents from the given URL.

    Returns:
        docs: The loaded documents from the URL.
"""


class CustomDocumentLoader:

    docs: list = []

    def __init__(self, loader=WebBaseLoader):
        self.loader = loader

    @classmethod
    def get_docs(cls):
        return cls.docs

    def load(self, url: str):
        """
        Loads the documents from the given URL.
        """
        if not url:
            logging.error(f"""No URL provided """)
            raise Exception(f"""No URL provided""")
        try:
            start_time = datetime.datetime.now()
            bs4_strainer = bs4.SoupStrainer()
            loader = self.loader(
                web_paths=(url,),
                bs_kwargs={"parse_only": bs4_strainer},
            )
            docs = loader.load()
            CustomDocumentLoader.docs = docs
            logging.info(f"""Loaded documents from {url}, page content: {len(
                CustomDocumentLoader.docs[0].page_content)}, Time Taken: {time_taken(start_time)} \n""")
        except Exception as e:
            logging.error(f"""Failed to load documents from {
                          url}: {str(e)} \n""")
            raise Exception(f"""Failed to load documents from {
                url}: {str(e)}""")
