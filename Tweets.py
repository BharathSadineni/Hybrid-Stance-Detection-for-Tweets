import re
import emoji
import time
import requests
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Optional, Dict
import jmespath

# Define emoji pattern
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "]+"
)

from playwright.sync_api import sync_playwright


# Retry decorator with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def scrape_and_clean_tweet(url: str) -> Optional[str]:
    _xhr_calls = []

    def intercept_response(response):
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    with sync_playwright() as pw:
        try:
            # Check if Playwright browsers are installed
            pw.chromium
        except Exception as e:
            raise EnvironmentError(
                "Playwright browsers are not installed. Run 'playwright install' to download them."
            )

        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        page.on("response", intercept_response)
        page.goto(url)

        # Check if the URL corresponds to a tweet
        if "twitter.com" not in page.url:
            raise ValueError("The provided URL is not a tweet.")

        # Check if the tweet exists
        if "error" in page.url:
            raise ValueError("The tweet does not exist or is not accessible.")

        page.wait_for_selector("[data-testid='tweet']")

        tweet_calls = [f for f in _xhr_calls if "TweetResultByRestId" in f.url]
        if not tweet_calls:
            raise ValueError("No valid tweet data found.")

        # Collect all text from tweet calls
        full_text = ""
        for xhr in tweet_calls:
            data = xhr.json()
            tweet_text = parse_tweet(data["data"]["tweetResult"]["result"])
            full_text += tweet_text + "\n"  # Concatenate tweet text

        return (
            full_text.strip()
        )  # Return the full text, stripped of leading/trailing whitespace

    return None  # Return None if scraping fails or no text is found


def parse_tweet(data: Dict) -> str:
    try:
        result = jmespath.search(
            """{
            text: legacy.full_text
        }""",
            data,
        )

        text = result.get("text", "")

        # Remove emojis
        text = emoji_pattern.sub("", text)

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove hashtags and mentions
        text = re.sub(r"#[^\s]+", "", text)
        text = re.sub(r"@[^\s]+", "", text)

        # Remove extra spaces
        text = " ".join(text.split())

        if not text:
            raise ValueError("The tweet does not contain clean text.")

        return text

    except Exception as e:
        raise ValueError(f"Error parsing tweet data: {str(e)}")


if __name__ == "__main__":
    try:
        tweet_url = input("Enter the URL of the tweet: ")
        cleaned_text = scrape_and_clean_tweet(tweet_url)
        print(f"Cleaned Text: {cleaned_text}")

    except ValueError as e:
        print(e)
    except EnvironmentError as e:
        print(e)
    except RequestException as e:
        print("An error occurred while fetching the tweet:", str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
