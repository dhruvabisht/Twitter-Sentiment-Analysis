#CREATING THE IMPORTS AND GETTING ALL REQUIRED LIBRARIES

from tkinter import *
import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

#NOW INITIALISING THE ROOT VARIABLE FOR THE GUI LOOP
root = Tk()

#Creating a function to run the command from the console
def Twitter():    
    print("\t\t\t\tTHIS IS THE SOCIAL MEDIA SENTIMENTAL ANALYSIS RESULT SCREEN")
    import warnings
    warnings.filterwarnings("ignore")
    #Now I have taken two Strings as Sample and removed the punctuations from them
    #Converted them in string as tokens in an array to shocase how my program actually works
    import string
    print("\nPRINTING THE ENGLISH LANGUAGE PUNCTUATIONS : "+string.punctuation)
    Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'
    Test_punc_removed = []
    for char in Test: 
        if char not in string.punctuation:
            Test_punc_removed.append(char)
            
    print("\n\nTHE SAMPLE STRING WITHOUT PUNCTUATIONS : \n")
    print(Test_punc_removed)
    Test_punc_removed_join = ''.join(Test_punc_removed)
    print("\n\nTHE SAMPLE STRING JOINED TOGETHER : \n")
    print(Test_punc_removed_join)

    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    print("\n\nPRINTING THE STOP WORDS IN ENGLISH LANGUAGE : \n")
    print(stop_words)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    print("\n\nTHE SAMPLE STRING IN FORM OF AN INTEGER ARRAY WITH WORDS AS TOKENS : \n")
    print(Test_punc_removed_join_clean)
    Test_punc_removed_join_clean_joined = ''.join(Test_punc_removed_join_clean)
    print("\n\nSAMPLE DATA WITH ONLT TEXT : \n")
    print(Test_punc_removed_join_clean_joined)

    from sklearn.feature_extraction.text import CountVectorizer
    sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sample_data)
    print("\n\nTHE SAMPLE DATA PROCESSED : \n")
    print(vectorizer.get_feature_names())
    print(X.toarray())
    print("\nPLEASE WAIT A FEW SECONDS\nGETTING THE TWEETS FROM TWITTER SERVER.........")
    
    #Importing the Configurations
    twitterApiKey='YOUR_TWITTER_API_KEY'
    twitterApiSecret='YOUR_TWITTER_API_SECRET'
    twitterApiAccessToken='YOUR_TWITTER_API_ACESS_TOKEN'
    twitterApiAccessTokenSecret='YOUR_TWITTER_API_ACESS_TOKEN_SECRET'

    #Authenticating My Keys and Tokens
    auth=tweepy.OAuthHandler(twitterApiKey,twitterApiSecret)
    auth.set_access_token(twitterApiAccessToken,twitterApiAccessTokenSecret)
    twitterApi = tweepy.API(auth,wait_on_rate_limit=True)    
    tweets = tweepy.Cursor(twitterApi.user_timeline, screen_name = twitterAccount.get(),count=None,since_id=None,max_id=None,trim_user=True,exclude_replies=True,contributor_details=False, include_entities=False).items(1000);

    #Creating a Datframe using the Tweets 
    df = pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweet'])
    print("\n\nPRINTING THE COLLECTION OF TWEETS : \n")
    print(df.head())
    
    #Function to convert the Entire Collection of Tweets into a single String
    def message_cleaning(message):
        Test_punc_removed = [char for char in message if char not in string.punctuation]
        Test_punc_removed_join = ''.join(Test_punc_removed)
        Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
        return Test_punc_removed_join_clean

    #Function to Remove Mentions, Retweets , Urls and Hastags 
    def Clean(text):    
            text = re.compile('\#').sub('', re.compile('RT @').sub('@', str(text), count=1).strip())
            text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str(text)).split())
            return text
    df['Tweet']=df['Tweet'].apply(Clean)
    df = df.drop(df[df['Tweet']==''].index)
    twitter_df_clean = df['Tweet'].apply(message_cleaning)
    twitter_df_clean