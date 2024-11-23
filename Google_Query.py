import re
from typing import List
from googlesearch import search
import requests
from bs4 import BeautifulSoup


# Define a function to perform a Google search and return a list of URLs
def google_search_keywords(keywords: List[str]) -> List[str]:
    query = " ".join(keywords)
    urls = list(search(query, num=50, stop=50))
    return urls


# Define a function to check if a URL is likely to contain an article
def is_article_url(url: str) -> bool:
    # Common patterns found in article URLs
    article_patterns = ["/news/", "/article/", "/blog/", "/post/", "/story/"]
    return any(pattern in url.lower() for pattern in article_patterns)


# Define a function to clean text
def clean_text(text: str) -> str:
    # Remove HTML tags and attributes
    text = re.sub(r"<[^>]+>", "", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


# Define a function to get article text from a URL with a word limit
def get_article_text(url: str, word_limit: int = 1000) -> str:
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Raise an exception for non-200 status codes
        response.raise_for_status()
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        # Find all paragraphs in the HTML
        paragraphs = soup.find_all("p")
        # Concatenate the text within paragraphs into a single string
        article_text = " ".join(paragraph.get_text() for paragraph in paragraphs)
        # Clean the article text
        cleaned_text = clean_text(article_text.strip())
        # Limit the text to the specified word limit
        words = cleaned_text.split()[:word_limit]
        return " ".join(words)
    except Exception as e:
        # Return an empty string if there's an error fetching or processing the text
        print(f"Failed to fetch or process article text from {url}: {str(e)}")
        return ""


if __name__ == "__main__":
    try:
        # Input keywords from the user
        input_keywords = input("Enter keywords separated by commas: ").split(",")
        # Perform a Google search with the keywords and get a list of URLs
        urls = google_search_keywords(input_keywords)

        print("Search results:")
        # Iterate over the URLs
        for i, url in enumerate(urls, start=1):
            # Check if the URL is likely to contain an article
            if is_article_url(url):
                print(f"{i}. {url}")
                # Get the article text from the URL and print it with a word limit
                article_text = get_article_text(url)
                print("Text extracted (up to 1000 words):")
                print(article_text)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
