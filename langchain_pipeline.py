import datetime
import os
import logging
from dotenv import load_dotenv

from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils.ai.custom_document_loader import CustomDocumentLoader
from utils.ai.custom_document_retriever import CustomDocumentRetriever
from utils.ai.custom_document_splitter import CustomDocumentSplitter
from utils.ai.custom_embedding_store import CustomEmbeddingStore
from utils.helpers import check_if_model_exists, time_taken


try:
    logging.info(f"""Loading environment variables \n""")
    load_dotenv(dotenv_path='.env.local')
    logging.info(f"""Environment variables loaded \n""")
except Exception as e:
    logging.error(f"""Failed to load environment variables: {str(e)} \n""")
    raise Exception(f"""Failed to load environment variables: {str(e)}""")


class LangchainPipeline:
    def __init__(
        self,
        prompt: str = None,
        url: str = None,
        contextualized_q_system_prompt: str = None,
        system_prompt: str = None,
        qa_prompt: str = None,
        *args,
        **kwargs
    ):
        """
        Initializes the LangchainPipeline with optional configurations.

        :param prompt: The prompt template to be used in the pipeline.
        :param url: The URL from which to load the documents.
        :param contextualized_q_system_prompt: The prompt for contextualized Q&A.
        :param system_prompt: The system prompt used in conversation.
        :param qa_prompt: The prompt specifically for question answering.
        """

        # Initialize internal state
        self.loader = CustomDocumentLoader()
        self.splitter = CustomDocumentSplitter(doc_loader=self.loader or None)
        self.embedding_store = CustomEmbeddingStore(
            doc_splitter=self.splitter or None)
        self.document_retriever = CustomDocumentRetriever(
            embedding_store=self.embedding_store or None)

        self.__llm = None
        self.__rag_chain = None
        self.__retriever = None
        self.__chat_store: dict = {}
        self.__conversation_chain = None

        # Initialize configuration variables from kwargs or defaults
        self.__prompt = prompt
        self.__qa_prompt = qa_prompt
        self.__url = url
        self.__contextualized_q_system_prompt = contextualized_q_system_prompt
        self.__system_prompt = system_prompt

    def setup_environment_variables(self):
        """
        Set up the environment variables for langsmith
        """
        try:
            os.environ["LANGCHAIN_TRACING_V2"] = os.getenv(
                "LANGCHAIN_TRACING_V2")
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
            os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
            logging.info(f"""Fetched langsmith API key \n""")
        except Exception as e:
            logging.error(f"""Failed to fetch langsmith API key: {
                          str(e)} \n""")
            raise Exception(f"""Failed to fetch langsmith API key: {
                str(e)}""")

    def setup(self, url, model_name, qa_chain=True):
        """
        Sets up the pipeline environment and initializes components.
        """
        start_time = datetime.datetime.now()
        self.setup_environment_variables()
        # self.__url = url
        try:
            try:
                self.setup_retriever(url)
            except Exception as e:
                logging.error(f"""Failed to set up retriever: {
                              str(e)} """)
                raise Exception(f"""Failed to set up retriever: {str(e)}""")
            try:
                self.setup_llm(model_name)
            except Exception as e:
                logging.error(f"""Failed to set up LLM: {
                              str(e)} """)
                raise Exception(f"""Failed to set up LLM: {str(e)}""")
            try:
                self.setup_rag_chain()
            except Exception as e:
                logging.error(f"""Failed to set up RAG chain: {
                              str(e)} """)
                raise Exception(f"""Failed to set up RAG chain: {str(e)}""")
            if qa_chain:
                logging.debug(
                    f"Setting up conversation RAG chain \n")
                try:
                    self.setup_conversation_chain()
                except Exception as e:
                    logging.error(f"""Failed to set up conversation chain: {
                                  str(e)} """)
                    raise Exception(f"""Failed to set up conversation chain: {
                        str(e)}""")
            logging.info(f"""Pipeline setup complete, Time Taken: {
                time_taken(start_time)} \n""")
        except Exception as e:
            logging.error(f"""Failed to set up langchain environment: {
                          str(e)}  \n""")
            raise Exception(
                f"""Failed to set up langchain environment: {str(e)}""")

    def setup_retriever(self, url):
        """
        Sets up the document retriever.
        """
        # if self.__url == url:
        #     logging.info(f"Retriever already set up for {url} ")
        #     return
        try:
            try:
                self.loader.load(url)
            except Exception as e:
                logging.error(f"""Failed to load documents: {
                              str(e)} """)
                raise Exception(f"""Failed to load documents: {str(e)}""")
            try:
                self.splitter.split()
            except Exception as e:
                logging.error(f"""Failed to split documents: {
                              str(e)} """)
                raise Exception(f"""Failed to split documents: {str(e)}""")
            try:
                self.embedding_store.store_embeddings()
            except Exception as e:
                logging.error(f"""Failed to store documents: {
                              str(e)} """)
                raise Exception(f"""Failed to store documents: {str(e)}""")
            try:
                self.document_retriever.retrieve()
            except Exception as e:
                logging.error(f"""Failed to retrieve documents: {
                              str(e)} """)
                raise Exception(f"""Failed to retrieve documents: {str(e)}""")
            try:
                self.__retriever = self.document_retriever.get_retriever()
            except Exception as e:
                logging.error(f"""Failed to get retriever: {
                              str(e)} """)
                raise Exception(f"""Failed to get retriever: {str(e)}""")
            self.__url = url
        except Exception as e:
            logging.error(f"""Failed to set up retriever: {
                str(e)} """)
            raise Exception(f"""Failed to set up retriever: {str(e)}""")

    def setup_llm(self, model_name):
        """
        Initializes the language model.
        """
        try:
            if not check_if_model_exists(model_name):
                logging.error(f"Model {model_name} does not exist ")
                raise Exception(f"Model {model_name} does not exist")
            self.__llm = ChatOllama(model=model_name)
            logging.info(f"\nInitialized LLM \n")
        except Exception as e:
            logging.error(f"Failed to set up LLM: {str(e)} ")
            raise Exception(f"Failed to set up LLM: {str(e)}")

    def setup_rag_chain(self):
        """
        Sets up the RAG chain for retrieval-augmented generation.
        """
        # function to format the documents
        def format_docs(docs):
            start_time = datetime.datetime.now()
            logging.info(f"Formatting documents \n")
            formatted_docs = "\n\n".join(doc.page_content for doc in docs)
            logging.debug(f"Time taken to format documents: {
                          time_taken(start_time)}\n")
            return formatted_docs
        logging.debug(f'retriever = {type(self.__retriever)}, format_docs = {
                      type(format_docs)}, prompt = {type(self.__prompt)}, llm = {type(self.__llm)}\n')
        # Ensure the retriever and other components are initialized
        if not self.__retriever:
            raise Exception(
                "Retriever not set up. Please set up retriever before RAG chain.")
        if not self.__prompt:
            raise Exception(
                "Prompt not set up. Please set up prompt before RAG chain.")
        if not self.__llm:
            raise Exception(
                "LLM not set up. Please set up LLM before RAG chain.")
        try:
            self.__rag_chain = (
                {"context": self.__retriever | format_docs,
                    "question": RunnablePassthrough()}
                | PromptTemplate.from_template(self.__prompt)
                | self.__llm
                | StrOutputParser()
            )
            logging.info(f"Chain ready to generate response ")
        except Exception as e:
            logging.error(f"""Failed to set up RAG chain: {
                          str(e)} """)
            raise Exception(f"Failed to set up RAG chain: {str(e)}")

    def setup_conversation_chain(self):
        """
        Sets up a conversation chain that maintains context across interactions.
        """
        # function to trim and summarize the conversation history
        def trim_and_summarize_history(chat_history: BaseChatMessageHistory) -> BaseChatMessageHistory:
            try:
                # keep the latest 25 messages
                latest_messages = chat_history.messages[-25:]
                logging.info(f""" keeping latest {
                             len(latest_messages)} messages """)
                # divide the remaining messages into 5 parts
                remaining_messages = chat_history.messages[:-25]
                part_size = len(remaining_messages) // 5
                parts = [remaining_messages[i:i+part_size]
                         for i in range(0, len(remaining_messages), part_size)]

                # summarize each part using your text summarizing tool
                summarized_parts = []
                for part in parts:
                    summarized_part = self.generate(part)
                    summarized_parts.append(summarized_part)
                logging.info(f""" summarized {len(summarized_parts)} parts """)
                # add the summarized parts back to the conversation history
                chat_history.messages = latest_messages + summarized_parts
                logging.info(f"""Total messages in chat history: {
                    len(chat_history.messages)} """)
                return chat_history
            except Exception as e:
                logging.error(f"""Failed to trim and summarize history: {
                              str(e)} """)
                raise Exception(f"""Failed to trim and summarize history: {
                    str(e)}""")

        # function to get the session history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.__chat_store:
                self.__chat_store[session_id] = ChatMessageHistory()
            chat_history = self.__chat_store[session_id]
            if (len(chat_history.messages) > 50):
                logging.info(
                    f""" chat history is too long, trimming and summarizing """)
                chat_history = trim_and_summarize_history(chat_history)
            return chat_history

        try:
            # Set up the contextualized conversation chain prompt
            logging.info(f"""setting contextualized prompt """)
            contextualized_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.__contextualized_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
            # create the history aware retriever
            logging.info(f"""creating history aware retriever""")
            history_aware_retriever = create_history_aware_retriever(
                self.__llm,
                self.__retriever,
                contextualized_q_prompt,
            )
            # set up the QA prompt
            logging.info(f"""setting up QA prompt """)
            self.__qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.__system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
            # set up the QA chain
            logging.info(f"""creating QA chain """)
            qa_chain = create_stuff_documents_chain(
                self.__llm, self.__qa_prompt)
            # set up the retriever chain
            logging.info(f"""creating retriever chain """)
            rag_chain = create_retrieval_chain(
                history_aware_retriever,
                qa_chain,
            )
            # set up the conversation chain with the history
            logging.info(f"""setting up conversation chain """)
            self.__conversation_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            logging.info(f"""Conversation chain ready """)

        except Exception as e:
            logging.error(f"""Failed to set up conversation chain: {
                          str(e)} """)
            raise Exception(f"""Failed to set up conversation chain: {
                str(e)}""")

    def generate(self, url, query):
        """
        Generates a response using the RAG chain.
        """
        if not self.check_if_setup():
            logging.error(f"Setting pipeline ")
            self.setup(url, False)
        try:
            start_time = datetime.datetime.now()
            result = self.__rag_chain.invoke(query)
            logging.info(f"""Summary Generated, Time Taken: {
                         time_taken(start_time)} """)
            return result
        except Exception as e:
            logging.error(f"""Failed to generate response: {
                          str(e)} """)
            raise Exception(f"""Failed to generate response: {
                str(e)}""")

    def generate_response_from_conversation(self, query, session_id):
        """
        Generates a response from a conversation chain.
        """
        if not self.check_if_setup():
            logging.error(f"Setting pipeline ")
            self.setup(self.__url, True)
        try:
            logging.info(
                f"""Generating response from conversation """)
            start_time = datetime.datetime.now()
            result = self.__conversation_chain.invoke(
                {"input": query},
                config={
                    "configurable":
                        {"session_id": session_id}
                }
            )
            logging.info(f"""Response Generated, Time Taken: {
                         time_taken(start_time)} """)
            return result["answer"]
        except Exception as e:
            logging.error(f"Failed to generate response: {
                          str(e)} ")
            raise Exception(f"Failed to generate response: {str(e)}")

    def generate_streaming_response_from_conversation(self, query, session_id):
        """
        Generates a response from a conversation chain in a streaming manner.
        """
        if not self.check_if_setup():
            logging.info(f"Setting pipeline ")
            self.setup(self.__url, True)

        try:
            logging.info(
                f"Generating response from conversation ")
            start_time = datetime.datetime.now()

            # Use a generator to yield response chunks
            for chunk in self.__conversation_chain.stream(
                {"input": query},
                config={
                    "configurable": {"session_id": session_id}
                }
            ):
                if "answer" in chunk:
                    yield chunk["answer"]

        except Exception as e:
            logging.error(f"""Failed to generate streaming response: {
                          str(e)} """)
            raise Exception(f"Failed to generate streaming response: {str(e)}")

    def check_if_setup(self):
        if self.__rag_chain is None and self.__conversation_chain is None:
            return False
        return True

    def teardown(self):
        """
        Cleans up the pipeline environment.
        """
        self.__docs = []
        self.__docs_splits = None
        self.__vectorstore = None
        self.__retriever = None
        self.__llm = None
        self.__rag_chain = None
        self.__chat_store = {}
        self.__conversation_chain = None
        logging.info(f"Pipeline cleaned up ")


# system_prompt = (
#     """You are an assistant for question-answering tasks.
#     Use the following pieces of context to answer the question.
#     If you don't know the answer, just say that you don't know, don't try to make up an answer.
#     Use three sentences maximum and keep the answer as concise as possible.
#     Generate the response in markdown format wherever possible.
#     Always start your answer with "thanks for asking!".
#     \n\n
#     {context}
#     """
# )

# contextualized_q_system_prompt = (
#     """ Give a chat history and the latest user question
#         which might reference context from the chat history.
#         formulate a standalone question that can be understood
#         without the context of the chat history. Do not answer the question,
#         just reformulate it if necessary and otherwise pass it through.
#     """
# )

# qa_chain = True

# prompt_template = """Use the following pieces of context and provide the summary of it.
# Make sure to keep the summary concise and to the point.
# If you are unable to generate the summary, please let me know.
# {context}
# Question: {question}
# Helpful Answer:"""
# url = 'https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/'

# pipeline = LangchainPipeline(
#     prompt=prompt_template,
#     contextualized_q_system_prompt=contextualized_q_system_prompt,
#     system_prompt=system_prompt,
# )
# try:
#     pipeline.setup(url=url)
# except Exception as e:
#     logging.error(f"Failed to set up pipeline: {str(e)} ")
#     raise Exception(f"Failed to set up pipeline: {str(e)}")

# try:
#     query = input("Enter your query: ")
#     if qa_chain:
#         response = pipeline.generate_response_from_conversation(
#             query, "session_1")
#     else:
#         response = pipeline.generate(query)
#     logging.info(f"Response: {response} ")
# except Exception as e:
#     logging.error(f"Failed to generate response: {str(e)} ")
