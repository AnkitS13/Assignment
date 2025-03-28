### Nebula9.ai Lead Gen Chatbot and Web Scraper

This repository contains two Python scripts designed to work together to build a chatbot and scrape web content:

scrape_url.py: A web scraper that fetches and filters URLs from the Nebula9.ai website.

prompt_routing.py (main file) : A chatbot powered by LangChain, Chroma DB, and an LLM to handle queries about Nebula9.ai and related use cases.

wbl_demo.ipynb : A basic demo of the RAG pipeline explicitly without any prompt_routing or intent classification.
Prerequisites

Ensure you have the following installed:

Python 3.8+
Required Python libraries (install using pip):

pip install -r requirements.txt


### Files Overview
#### 1. scrape_url.py
This file scrapes the Nebula9.ai website for URLs and filters the results.

Features:
- Scrapes all links from the homepage.
- Filters for valid, absolute URLs.
- Outputs the filtered list of URLs to the console.
Usage:
- Open the file and modify the url variable if you wish to scrape a different website.
- Run the file:
    python scrape_url.py

#### 2. prompt_routing.py
This file is a chatbot for Nebula9.ai, designed to:

- Answer questions about Nebula9.ai services.
- Handle general queries and provide AI-generated answers.
- Gather lead details for users interested in Nebula9.ai’s offerings.

Key Features:

- Web Scraping Integration: Processes web pages using URLs provided by scrape_url.py.
- Document Chunking: Splits web content into smaller chunks for better processing.
- Query Handling: Classifies user queries into company_specific, general, or lead_generation.
- Uses a Retrieval-Augmented Generation (RAG) pipeline to fetch relevant information.
- Lead Generation: Collects user details and stores them in a CSV file.

Usage:
- Ensure scrape_url.py has been run and the web_pages list in prompt_routing.py is updated with the scraped links.
- Run the chatbot: python prompt_routing.py

Interact with the chatbot via the terminal.
Additional Notes:
- The script uses the Chroma DB for vector storage and retrieval.
- Ensure the GROQ_API_KEY environment variable is set correctly for LLM functionality.


