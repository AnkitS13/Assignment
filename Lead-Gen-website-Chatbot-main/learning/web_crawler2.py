import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import html2text
import json
from text_to_doc import get_doc_chunks

def crawl_website(base_url, max_pages=10):
    """
    Crawls a website starting from the base URL and processes up to max_pages.

    Args:
        base_url (str): The starting URL for the crawler.
        max_pages (int): Maximum number of pages to crawl.

    Returns:
        list: A list of tuples containing text content and metadata for each page.
    """
    visited = set()
    to_visit = [base_url]
    pages_data = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
    }

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch {current_url}: {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            text, metadata = process_page(soup, current_url)

            pages_data.append((text, metadata))
            visited.add(current_url)

            # Find all links on the page and add them to the queue
            for link in soup.find_all("a", href=True):
                full_url = urljoin(base_url, link["href"])
                if full_url not in visited and base_url in full_url:  # Stay within the same website
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Error processing {current_url}: {e}")

    return pages_data

def process_page(soup, url):
    """
    Extracts text content and metadata from a BeautifulSoup object of a page.

    Args:
        soup (BeautifulSoup): Parsed HTML of the page.
        url (str): The URL of the page.

    Returns:
        tuple: Text content and metadata dictionary.
    """
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.extract()

    # Extract text
    html2text_instance = html2text.HTML2Text()
    html2text_instance.images_to_alt = True
    html2text_instance.body_width = 0
    html2text_instance.single_line_break = True
    text = html2text_instance.handle(str(soup))

    # Extract metadata
    title = soup.title.string.strip() if soup.title else url.split('/')[-1]
    meta_description = soup.find("meta", attrs={"name": "description"})
    description = meta_description.get("content") if meta_description else title
    metadata = {
        'title': title,
        'url': url,
        'description': description
    }
    return text, metadata

# def save_to_json(data, output_file="webcrawler_scraped_data.json"):
#     """
#     Saves data to a JSON file.

#     Args:
#         data (list): List of dictionaries to save.
#         output_file (str): Filepath for the output JSON file.
#     """
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

# Crawl the website and retrieve all pages' data
base_url = "https://nebula9.ai/"
pages_data = crawl_website(base_url, max_pages=10)

# Process each page into document chunks and save them into JSON format
scraped_data = []

for text_content, metadata in pages_data:
    doc_chunks = get_doc_chunks(text_content, metadata)
    for idx, doc in enumerate(doc_chunks):
        # Prepare data for JSON
        scraped_data.append({
            "page_url": metadata["url"],
            "title": metadata["title"],
            "text": doc.page_content,
            "chunk_index": idx,
            "metadata": doc.metadata
        })

# Save to JSON file
# save_to_json(scraped_data, output_file="scraped_data.json")

# print("Scraped data has been saved to scraped_data.json.")
