import logging
import datetime
import json

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from langchain_pipeline import LangchainPipeline

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s: line:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize Flask application
app = Flask(__name__)
cors = CORS(app)


# Langchain Pipeline Setup
system_prompt = (
    """You are an assistant for question-answering tasks.
    Use the following pieces of context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Generate the response in markdown format wherever possible.
    Always end your answer with "Let me know if you want to know more!".
    \n\n
    {context}
    """
)

contextualized_q_system_prompt = (
    """Give a chat history and the latest user question
       which might reference context from the chat history.
       Formulate a standalone question that can be understood
       without the context of the chat history. Do not answer the question,
       just reformulate it if necessary and otherwise pass it through.
    """
)

qa_chain = True
prompt_template = """[INST]<<SYS>>Use the following pieces of context and provide the summary of it.
Make sure to keep the summary concise and to the point.
If you are unable to generate the summary, please let me know.<</SYS>>
{context}
Question: {question}
Helpful Answer:[/INST]"""

# Initialize Langchain Pipeline
pipeline = LangchainPipeline(
    prompt=prompt_template,
    contextualized_q_system_prompt=contextualized_q_system_prompt,
    system_prompt=system_prompt,
)


@app.route('/setup', methods=['POST'])
def setup():
    if request.method == 'POST':
        try:
            logging.debug(f"request data: {request.get_data().decode()}")
            data = json.loads(request.get_data().decode())
            logging.debug(f"Data: {data}")
            url = data.get("url")
            model_name = data.get("model_name") or 'llama3.1'
            if not url:
                logging.error(f"Missing URL ")
                return jsonify({'error': 'Missing required parameters'}), 400
            pipeline.setup(url, model_name=model_name)
            return jsonify({'message': 'Setup successful'}), 200
        except Exception as e:
            return jsonify({'error': f"Error occurred: {e}"}), 500
    else:
        return jsonify({'error': 'Invalid request method'}), 405


@app.route('/teardown', methods=['POST'])
def teardown():
    if request.method == 'POST':
        try:
            pipeline.teardown()
            return jsonify({'message': 'Cleanup successful'}), 200
        except Exception as e:
            return jsonify({'error': f"Error occurred: {e}"}), 500
    else:
        return jsonify({'error': 'Invalid request method'}), 405

""" This endpoint is used to summarize the given text.
    The text is summarized using the pipeline and the response is streamed back to the client."""
@app.route('/summarize', methods=['POST'])
def summarize():
    
    logging.debug(f"Request acknowledged ")
    
    # Check if the request method is POST and extract query and session_id
    # query: The text to summarize
    # session_id: The session id for the conversation
    
    if request.method == 'POST':
        try:
            data = json.loads(request.get_data().decode())
            query = data.get("query")
            session_id = data.get("session_id")

            if not query or not session_id:
                logging.error(
                    f"Missing query, or session_id ")
                return jsonify({'error': 'Missing required parameters'}), 400

        except Exception as e:
            logging.error(f"""Error occurred while reading request object: {
                          str(e)} """)
            return jsonify({'error': 'JSON object read failed'}), 400

        logging.info(f"Query Received: {query} ")

        # a generator function to stream the response back to the client
        def generate_response():
            try:
                logging.info(f"Streaming response started ")
                start_time = datetime.datetime.now()
                # Use the pipeline to generate the response incrementally
                for chunk in pipeline.generate_streaming_response_from_conversation(
                        query=query, session_id=session_id):
                    yield chunk.encode('utf-8')
                    logging.info(f"Yielded chunk: {chunk} ")
            except Exception as e:
                logging.error(f"""Error occurred during streaming: {
                              str(e)} """)
                yield f"Error: {str(e)}".encode('utf-8')
            finally:
                logging.info(f"""Streaming completed, Time Taken: {
                             (datetime.datetime.now() - start_time).total_seconds():.3f}s""")

        return Response(generate_response(), content_type='text/plain')
    else:
        return jsonify({'error': 'Invalid request method'}), 405


@app.route('/health')
def health():
    logging.debug(f'Health endpoint called at {datetime.datetime.now()}')
    return 'Ok'


@app.route('/')
def hello_world():
    return 'Text Summarization API'


if __name__ == '__main__':
    app.run(debug=True)
