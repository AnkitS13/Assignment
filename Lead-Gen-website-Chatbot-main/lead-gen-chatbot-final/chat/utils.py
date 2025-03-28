import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text):
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_web_document(url_list):
    loader = WebBaseLoader(web_paths = url_list)
    return loader.load()

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splits = []

    for doc in docs:
        doc_chunk = text_splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunk):
            chunk.metadata['chunk_index'] = idx
            splits.append(chunk)
    return splits

def scrape(url = 'https://www.nebula9.ai/'):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    web_pages = []
    try:
        reqs = requests.get(url, headers=headers, timeout=10)
        reqs.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Parse the page content
        soup = BeautifulSoup(reqs.text, 'html.parser')

        # Extract links and filter them
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                # Resolve relative URLs to absolute URLs
                absolute_url = urljoin(url, href)

                # Parse the URL and validate
                parsed = urlparse(absolute_url)
                if parsed.scheme in ['http', 'https'] and parsed.netloc:
                    web_pages.append(absolute_url)

        # Remove duplicates
        filtered_web_pages = list(set(web_pages))
        return filtered_web_pages

    except requests.exceptions.RequestException as e:
        return {'error': str(e)}