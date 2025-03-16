import requests
from bs4 import BeautifulSoup
import logging
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import nltk

# nltk.download()  # For sentence tokenization
# nltk.download('averaged_perceptron_tagger')  # For POS tagging

class WebScraper:
    def __init__(self, url):
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
        }

    def scrape_url(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            soup = BeautifulSoup(response.text, 'lxml')

            # Removing scripts, styles, footers, navigations, and buttons
            for element in soup(['script', 'style', 'footer', 'nav', 'button']):
                element.decompose()

            # Extract meaningful content from defined chunks
            content = soup.find('body')
            if not content:
                logging.warning("No body content found")
                return ""

            content_text = content.get_text(separator=' ', strip=True)
            chunks = self.semantic_chunking(content_text)
            
            return chunks

        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while making the request: {e}")
            return ""
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return ""

    def extract_chunks(self, content):
        # Define chunks using HTML5 structural elements and common content tags
        chunks = []
        # Including 'article', 'section', 'aside', and 'header' for broader structural parsing
        for tag in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'article', 'section', 'aside', 'header']):
            if tag.name in ['ul', 'ol']:
                # Treat entire lists as single chunks
                list_items = tag.find_all('li')
                list_text = ' '.join(li.get_text(separator=' ', strip=True) for li in list_items)
                chunks.append(list_text)
            elif tag.name in ['article', 'section', 'aside', 'header']:
                # Treat entire HTML5 structural elements as single chunks
                section_text = tag.get_text(separator=' ', strip=True)
                if section_text:
                    chunks.append(section_text)
            else:
                text = tag.get_text(separator=' ', strip=True)
                if text:
                    chunks.append(text)
        return chunks
    
    def semantic_chunking(self,content):
        sentences = sent_tokenize(content)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            # Example heuristic: start a new chunk on certain POS tags or sentence patterns
            if tagged[0][1] in ['NN', 'NNP']:  # Noun or proper noun at the beginning of the sentence
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                else:
                    current_chunk.append(sentence)
            else:
                current_chunk.append(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks



# Example usage:
# scraper = WebScraper('https://example.com')
# final_summary = scraper.scrape_url()
# print(final_summary)
