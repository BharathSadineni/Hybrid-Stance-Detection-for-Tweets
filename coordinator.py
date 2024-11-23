from Tweets import scrape_and_clean_tweet
from keyword_algorithm import extract_keywords
from Google_Query import google_search_keywords, is_article_url, get_article_text
from stac import preprocess_text, get_features, build_vectorizers, predict_stance
import numpy as np
import joblib


def main():
    try:
        # Load pre-trained models and vectorizers
        classifier = joblib.load("model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        vectorizer = joblib.load("vectorizer.pkl")

        # Default tweet URL
        default_tweet_url = "https://twitter.com/brfootball/status/1779979257944371539"
        print(f"The default tweet URL to be checked is: {default_tweet_url}")
        user_input = (
            input("Do you want to enter a new URL? Type 'yes' or 'no': ")
            .strip()
            .lower()
        )

        # Determine which URL to use based on user input
        if user_input == "yes":
            tweet_url = input("Please enter the new tweet URL: ")
        else:
            tweet_url = default_tweet_url

        cleaned_text = scrape_and_clean_tweet(tweet_url)
        keywords = extract_keywords(cleaned_text)

        urls = google_search_keywords(keywords)
        article_texts = [get_article_text(url) for url in urls if is_article_url(url)]

        predicted_stances = predict_stance(
            " ".join(keywords), article_texts, classifier, label_encoder, vectorizer
        )

        stance_counts = np.array(
            [
                np.count_nonzero(predicted_stances == st)
                for st in ["agree", "disagree", "discuss", "unrelated"]
            ]
        )

        if stance_counts.sum() == 0:
            print("No stances were predicted.")
        else:
            # Calculating accuracy
            true_percentage = stance_counts[0] / stance_counts.sum() * 100

            # Checking if true percentage is above a certain threshold
            if true_percentage >= 60:
                tweet_status = "True"
            elif 50 <= true_percentage < 60:
                tweet_status = "Discuss"
            else:
                tweet_status = "False"

            print(
                f"Stance Summary: The Tweet is {tweet_status}. Percentage accuracy: {true_percentage:.2f}%"
            )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
