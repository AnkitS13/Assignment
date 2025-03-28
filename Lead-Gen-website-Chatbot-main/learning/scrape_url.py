import requests
from bs4 import BeautifulSoup

# URL to scrape
url = 'https://nebula9.ai/'

# Mimic a browser with headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}

try:
    # Send GET request
    reqs = requests.get(url, headers=headers, timeout=10)
    reqs.raise_for_status()  # Raise HTTPError for bad responses

    # Parse the page content
    soup = BeautifulSoup(reqs.text, 'html.parser')

    # Print the entire page content for inspection
    #print("Page Content:")
    #print(soup.prettify())

    # Extract and print all links
    print("\nExtracted Links:")
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            print(href)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
