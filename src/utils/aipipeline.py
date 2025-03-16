from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import datetime
# import tensorflow as tf
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

import nltk
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
from nltk.corpus import stopwords

# Setup to disable GPUs for TensorFlow if needed
# physical_devices = tf.config.list_physical_devices('GPU')

# try:
#     tf.config.set_visible_devices([], 'GPU')
# except:
#     pass  # Handle exceptions if the GPUs cannot be disabled


logging.basicConfig(filename='app.log', level=logging.DEBUG)

def log_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class AIPipeline:
    def __init__(self) -> None:
        logging.info(f"Initializing AI Pipeline {log_time()}")
        self.checkpoint = 'facebook/bart-large-cnn'
        self.tokenizer = None
        self.model = None
        self.device = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
        # self.device = 'cpu'
        logging.info(f"Device: {self.device} {log_time()}")
        

    def compile_model(self):
        start_time = datetime.datetime.now()
        if self.tokenizer and self.model:
            logging.info(f"Model and tokenizer already compiled {log_time()}")
            return

        logging.info(f"Compiling model with checkpoint: {self.checkpoint} {log_time()}")
        try:
            self.tokenizer = BartTokenizer.from_pretrained(self.checkpoint)
            self.model = BartForConditionalGeneration.from_pretrained(self.checkpoint).to(self.device)
            logging.info(f"Model compiled successfully {log_time()}")
        except Exception as e:
            logging.error(f"Failed to compile model: {str(e)} {log_time()}")
            raise e
        finally:
            logging.info(f"Time taken to compile model: {(datetime.datetime.now() - start_time).total_seconds():.3f}s")

    def preprocess_text(self, text):
        # Example of text preprocessing (optional)
        logging.debug(f"Preprocessing text at {log_time()}")
        text = text.strip()  # Basic trimming
        # Additional preprocessing can be added here
        return text
    
    def post_process_summary(self,summary):
        # Tokenize sentences
        sentences = sent_tokenize(summary)
        processed_sentences = []

        # Stopwords list
        stop_words = set(stopwords.words('english'))

        for sentence in sentences:
            # Tag parts of speech
            words = nltk.word_tokenize(sentence)
            tagged_words = pos_tag(words)
            
            # Remove redundant or overly common words (simple example)
            filtered_sentence = [word for word, tag in tagged_words if word.lower() not in stop_words and tag.startswith('NN')]
            
            # Reconstruct the sentence - this example simply joins the filtered words
            processed_sentences.append(' '.join(filtered_sentence))

        # Rebuild the processed summary
        processed_summary = ' '.join(processed_sentences)
        return processed_summary

    def predict(self, text, max_length=20):
        # logging.debug(f"Inference input: {text} at {log_time()}")
        if not self.model or not self.tokenizer:
            logging.error(f"Model or tokenizer not compiled {log_time()}")
            raise ValueError("Model or tokenizer not compiled")

        text = self.preprocess_text(text)

        inputs = self.tokenizer.encode(f'{text}', return_tensors='pt', max_length=1024, truncation=True).to(self.device)

        try:
            output = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=5,
                repetition_penalty=3.0,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                # temperature=0.7,
                # do_sample=True,
            )

            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # logging.debug(f"Decoded output: {decoded_output} at {log_time()}")

            return decoded_output

        except Exception as e:
            logging.error(f"Error during generation: {str(e)} {log_time()}")
            raise e
        
    def summarize_chunks(self, chunks, max_length=30):
        logging.info(f"Summarizing chunks at {log_time()}")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.predict, chunk, max_length): chunk for chunk in chunks}
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")
        return results

