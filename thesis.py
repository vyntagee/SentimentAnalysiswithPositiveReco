import re
import pandas as pd
from nltk.corpus import stopwords
import pickle
import tweepy as tw
import nltk
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))

negative_to_positive = {'not': 'great', 'bad': 'good', 'terrible': 'wonderful', 'enough': 'not enough'}
consumer_api_key = '3nVuSoBZnx6U4vzUxf5w'
consumer_api_secret = 'Bcs59EFbbsdF6Sl9Ng71smgStWEGwXXKSjYvPVt7qys'
auth = tw.OAuthHandler(consumer_api_key, consumer_api_secret)
api = tw.API(auth, wait_on_rate_limit=True)

def analyze_tweets(search_query):
    search_words = search_query
    date_since = "2019-03-6"
    date_until = "2023-12-6"

    tweets = tw.Cursor(api.search_tweets,
                       q=search_words,
                       since=date_since,
                       until=date_until
                       ).items(1000)

    tweets_copy = []
    for tweet in tqdm(tweets):
        tweets_copy.append(tweet)
    print(f"New tweets retrieved: {len(tweets_copy)}")

    data = pd.DataFrame()
    for tweet in tqdm(tweets_copy):
        hashtags = []
        try:
            for hashtag in tweet.entities["hashtags"]:
                hashtags.append(hashtag["text"])
        except:
            pass
        data = data.append(pd.DataFrame({'user_name': tweet.user.name,
                                          'date': tweet.created_at,
                                          'text': tweet.text,
                                          }, index=[0]))


    # Load the CSV file into a DataFrame and merge it with the existing datadf DataFrame
    new_df = pd.read_csv('New_Era_University13.csv')
    data = pd.concat([data, new_df], ignore_index=True)

    negative_to_positive = {'not': 'great', 'bad': 'good', 'terrible': 'wonderful', 'enough': 'not enough'}

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[#@]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    # Apply the preprocess_text function to each row of the data DataFrame
    data['text'] = data['text'].apply(preprocess_text)

    # Load the vectorizer and the classifier
    vectorizer = pickle.load(open('vectorizers.pkl', 'rb'))
    voting_classifier = pickle.load(open('voting_classifiers.pkl', 'rb'))

    # Transform the data using the vectorizer
    data_transformed = vectorizer.transform(data['text'])

    # Make predictions using the classifier
    predictions = voting_classifier.predict(data_transformed)

    # Add the predictions as a new column to the data DataFrame
    data['sentiment'] = predictions

    # Define the custom order of the sentiment categories
    order = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Sort the DataFrame by the 'sentiment' column using the custom order
    data_sorted = data.sort_values(by='sentiment', key=lambda x: x.map(order))

    LearningPlatform_keywords = ['#LearningPlatform','LearningPlatform','learningplatform','learningPlatform', 'NEUVLE', 'exam', 'quiz', 'seatwork','homework']
    Cashier_keywords = ['cashier', 'cash','payment']
    Registrar_keywords = ['registrar', 'account','email']

    data_sorted['category'] = 'general'
    for index, row in data_sorted.iterrows():
        tweet = str(row['text']).lower()
        if any(keyword in tweet for keyword in LearningPlatform_keywords):
            data_sorted.at[index, 'category'] = 'LearningPlatform'
        elif any(keyword in tweet for keyword in Cashier_keywords):
            data_sorted.at[index, 'category'] = 'Cashier'
        elif any(keyword in tweet for keyword in Registrar_keywords):
            data_sorted.at[index, 'category'] = 'Registrar'

    def modify_negative_text(text):
        words = text.split()
        for i, word in enumerate(words):
            if word in negative_to_positive.keys():
                words[i] = negative_to_positive[word]
        modified_text = ' '.join(words)
        return modified_text

    for index, row in data_sorted.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        category = row['category']
        if sentiment == 'negative':
            modified_text = modify_negative_text(text)
            if category == 'LearningPlatform':
                data_sorted.at[index, 'recommendation'] = f"Consider making improvements to the Learning Platform. Issues mentioned: {modified_text}"
            elif category == 'cashier':
                data_sorted.at[index, 'recommendation'] = f"Consider making improvements to the cashier service. Issues mentioned: {modified_text}"
            elif category == 'registrar':
                data_sorted.at[index, 'recommendation'] = f"Consider making improvements to the registrar service. Issues mentioned: {modified_text}"
            else:
                data_sorted.at[index, 'recommendation'] = f"General issues mentioned: {modified_text}"
        elif sentiment == 'positive':
            if category == 'LearningPlatform':
                data_sorted.at[index, 'recommendation'] = "Great job on the Learning Platform! Keep up the good work."
            elif category == 'cashier':
                data_sorted.at[index, 'recommendation'] = "Great job with the cashier service! Keep up the good work."
            elif category == 'registrar':
                data_sorted.at[index, 'recommendation'] = "Great job with the registrar service! Keep up the good work."
            else:
                data_sorted.at[index, 'recommendation'] = 'Keep up the good work in general!'
        else:
            data_sorted.at[index, 'recommendation'] = 'Neutral feedback, no specific recommendation.'

    results = data_sorted.to_dict(orient="records")
    return results