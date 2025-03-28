import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# URL to scrape
url = 'https://nebula9.ai/'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}

# List to store filtered links
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

    # Print the filtered links
    print("\nFiltered Links:")
    for page in filtered_web_pages:
        print(page)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
