import pandas as pd
import numpy as np
import nltk
import re
import os
import time
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

start_time = time.time()

# Load stopwords and stemmer once
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def build_vectorizers(X_train):
    print("Building and fitting vectorizer...")
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        max_features=2**14,
        ngram_range=(1, 2),
    )
    vectorizer.fit(X_train)  # Ensure vectorizer is fitted here
    print("Vectorizer built and fitted.")
    return vectorizer


def get_features(X, vectorizer):
    print("Generating features...")
    features = vectorizer.transform(tqdm(X, desc="Vectorizing data"))
    print("Features generated.")
    return features


def build_and_train_model(X_train, y_train, X_test, y_test):
    print("Checking for pre-existing model files...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    if not (
        os.path.exists("model.pkl")
        and os.path.exists("vectorizer.pkl")
        and os.path.exists("label_encoder.pkl")
    ):
        print("No existing model files found. Building and training model...")
        vectorizer = build_vectorizers(X_train)
        X_train_features = get_features(X_train, vectorizer)
        X_test_features = get_features(X_test, vectorizer)

        classifier = XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            max_depth=4,
            min_child_weight=5,
            reg_lambda=2,
            reg_alpha=1,
        )
        classifier.fit(X_train_features, y_train_encoded)
        joblib.dump(classifier, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
    else:
        print("Model files found. Loading existing model...")
        classifier = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        X_train_features = get_features(
            X_train, vectorizer
        )  # Needed for predictions below
        X_test_features = get_features(
            X_test, vectorizer
        )  # Needed for predictions below

    print("Using model to make predictions...")
    train_predictions = classifier.predict(X_train_features)
    test_predictions = classifier.predict(X_test_features)
    train_accuracy = accuracy_score(y_train_encoded, train_predictions)
    test_accuracy = accuracy_score(y_test_encoded, test_predictions)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(
        classification_report(
            y_test_encoded, test_predictions, target_names=label_encoder.classes_
        )
    )
    return classifier, label_encoder, vectorizer


# Data loading and preprocessing
print("Loading data...")
data = pd.read_csv("balanced_train_dataset_10000.csv")
X = data["Headline"] + " " + data["articleBody"]
y = data["Stance"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split completed. Starting model training...")
model, label_encoder, vectorizer = build_and_train_model(
    X_train, y_train, X_test, y_test
)

print("All processes completed successfully.")
print("Total compilation time: {:.2f} seconds".format(time.time() - start_time))


def predict_stance(headline, article_bodies, classifier, label_encoder, vectorizer):
    """
    Predicts the stances for a given headline against multiple article bodies using the specified ML model and components.

    Parameters:
    - headline (str): The headline for the predictions.
    - article_bodies (list of str): Each article body to predict against.
    - classifier (model): The trained classification model.
    - label_encoder (LabelEncoder): Encoder for transforming labels.
    - vectorizer (TfidfVectorizer): The vectorizer for text data.

    Returns:
    - list of str: The predicted stances for each article body.
    """
    if not isinstance(headline, str):
        raise ValueError("The 'headline' must be a string.")
    if not isinstance(article_bodies, list) or not all(
        isinstance(body, str) for body in article_bodies
    ):
        raise ValueError("The 'article_bodies' must be a list of strings.")

    combined_texts = [headline + " " + body for body in article_bodies]
    features = get_features(combined_texts, vectorizer)
    predictions = classifier.predict(features)
    predicted_stances = label_encoder.inverse_transform(predictions)
    return predicted_stances
