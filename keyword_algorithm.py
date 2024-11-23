from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string

# Ensure NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def extract_keywords(sentence: str) -> List[str]:
    # Load a set of stopwords
    stopwords_set = set(stopwords.words("english"))

    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Filter words to remove stopwords and punctuation, allow alphabetic and alphanumeric words
    keywords = [
        word.lower()
        for word in words
        if word.lower() not in stopwords_set
        and not all(char in string.punctuation for char in word)
        and (word.isalnum() or any(char.isdigit() for char in word))
    ]

    # Lemmatization using NLTK's WordNetLemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    keywords = [lemmatizer.lemmatize(word) for word in keywords]

    # Remove duplicates
    keywords = list(set(keywords))

    return keywords


if __name__ == "__main__":
    try:
        tweet_url = input("Enter the URL of the tweet: ")
        from Tweets import scrape_and_clean_tweet  # assuming this is defined elsewhere

        cleaned_text = scrape_and_clean_tweet(tweet_url)
        print(f"Cleaned Text: {cleaned_text}")

        keywords = extract_keywords(cleaned_text)

        if not keywords:
            print("No keywords extracted.")
        else:
            print("Keywords:", ", ".join(keywords))

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
