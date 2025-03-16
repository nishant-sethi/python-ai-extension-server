from langchain_pipeline import LangchainPipeline
import logging

system_prompt = (
    """You are an assistant for question-answering tasks.
    Use the following pieces of context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Generate the response in markdown format wherever possible.
    Always start your answer with "thanks for asking!".
    \n\n
    {context}
    """
)

contextualized_q_system_prompt = (
    """ Give a chat history and the latest user question
        which might reference context from the chat history.
        formulate a standalone question that can be understood
        without the context of the chat history. Do not answer the question,
        just reformulate it if necessary and otherwise pass it through.
    """
)

qa_chain = True

prompt_template = """Use the following pieces of context and provide the summary of it.
Make sure to keep the summary concise and to the point.
If you are unable to generate the summary, please let me know.
{context}
Question: {question}
Helpful Answer:"""
url = 'https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/'

pipeline = LangchainPipeline(
    prompt=prompt_template,
    contextualized_q_system_prompt=contextualized_q_system_prompt,
    system_prompt=system_prompt,
)
try:
    pipeline.setup(url=url, model_name="llama3.2", qa_chain= False)
except Exception as e:
    logging.error(f"Failed to set up pipeline: {str(e)} ")
    raise Exception(f"Failed to set up pipeline: {str(e)}")

try:
    query = input("Enter your query: ")
    if qa_chain:
        response = pipeline.generate_response_from_conversation(
            query, "session_1")
    else:
        response = pipeline.generate(query)
    logging.info(f"Response: {response} ")
except Exception as e:
    logging.error(f"Failed to generate response: {str(e)} ")
