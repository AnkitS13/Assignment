import requests
from bs4 import BeautifulSoup
import html2text

def get_data_from_website(url):
    """
    Retrieve text content and metadata from a given URL.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        tuple: A tuple containing the text content (str) and metadata (dict).
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
    }
    
    # Get response from the server
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch content: {response.status_code}")
        return None, None

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Removing js and css code
    for script in soup(["script", "style"]):
        script.extract()

    # Extract text in markdown format
    html = str(soup)
    html2text_instance = html2text.HTML2Text()
    html2text_instance.images_to_alt = True
    html2text_instance.body_width = 0
    html2text_instance.single_line_break = True
    text = html2text_instance.handle(html)

    # Extract page metadata
    try:
        page_title = soup.title.string.strip()
    except:
        page_title = url.split('/')[-1]
    meta_description = soup.find("meta", attrs={"name": "description"})
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    description = meta_description.get("content") if meta_description else page_title
    keywords = meta_keywords.get("content") if meta_keywords else ""

    metadata = {
        'title': page_title,
        'url': url,
        'description': description,
        'keywords': keywords
    }

    return text, metadata

# Example usage
url = "https://nebula9.ai/"
text_content, metadata = get_data_from_website(url)

if text_content and metadata:
    print("Text Content:\n", text_content[:3000])  # Print first 500 characters for brevity
    print("\nMetadata:\n", metadata)
else:
    print("Failed to retrieve data from the website.")