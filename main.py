import tweepy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import re

# ---- Twitter API keys (replace with your credentials) ---- #
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"
# ---------------------------------------------------------- #

# Authenticate Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

def clean_text(text):
    # Remove URLs, mentions, hashtags, special characters
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", text)
    text = text.lower()
    return text

def fetch_tweets(query, count=100):
    tweets = api.search_tweets(q=query, lang="en", count=count)
    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet.text)
    return tweet_list

def load_sample_data():
    # Load sample sentiment labelled data
    # You can replace this with your own dataset
    df = pd.read_csv("sample_data.csv")
    return df

def train_model(df):
    df['clean_text'] = df['text'].apply(clean_text)
    X = df['clean_text']
    y = df['sentiment']
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    # Save model and vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Model and vectorizer saved.")
    return model, vectorizer

def predict_sentiment(tweets, model, vectorizer):
    tweets_clean = [clean_text(t) for t in tweets]
    tweets_vec = vectorizer.transform(tweets_clean)
    preds = model.predict(tweets_vec)
    return list(zip(tweets, preds))

if __name__ == "__main__":
    # Load or train model
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        print("Loaded existing model and vectorizer.")
    except:
        print("Training new model...")
        df = load_sample_data()
        model, vectorizer = train_model(df)

    choice = input("Enter 1 to analyze tweets, 2 to train model: ")
    if choice == "1":
        query = input("Enter keyword or hashtag: ")
        tweets = fetch_tweets(query)
        results = predict_sentiment(tweets, model, vectorizer)
        for t, s in results[:10]:
            print(f"{s}: {t}")
        # Count sentiments
        sentiments = [s for t, s in results]
        print(pd.Series(sentiments).value_counts())
    else:
        print("Retraining model...")
        df = load_sample_data()
        train_model(df)
