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
    #Now we have taken two Strings as Sample and removed the punctuations from them
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

    #Getting Vector Values of tweets
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
    twitter_countvectorizer = vectorizer.fit_transform(df['Tweet'])
    print("\n\nTHE TWEETS IN FORM OF STRING TOKENS AND THE VECTOR ARRAY : \n")
    print(vectorizer.get_feature_names())
    print(twitter_countvectorizer.toarray())
    print("\n\nPRINTING THE COLLECTION OF TWEETS AFTER CLEANING : \n")
    print(df.head(10))

    #Function to get Ploarity of Tweets
    def getTextPolarity(txt):
        return TextBlob(txt).sentiment.polarity

    #Fuction to get the Subjectivity of Tweets
    def getTextSubjectivity(txt):
        return TextBlob(txt).sentiment.subjectivity

    #Applying the above functions to my Data Frame
    df['Subjectivity']=df['Tweet'].apply(getTextSubjectivity)
    df['Polarity']=df['Tweet'].apply(getTextPolarity)

    print("\n\nPRINTING THE DATA FRAME WITH SUBJECTIVITY AND POLARITY VALUES : \n")
    print(df.head(10))
    
    print("\n\nPRINTING THE DATA FRAME AFTER REMOVING THE NULL TWEETS : \n")
    df = df.drop(df[df['Tweet']==''].index)
    print(df.head(10))

    #Function to return the Score Values
    def getTextAnalysis(pol):
        if pol<0:
            return 0
        elif pol>0:
            return 1

    #Function to Assign labels to the tweets
    def assignLabels(pol):
        if pol<0:
            return 'Negative'
        elif pol==0:
            return 'Neutral'
        else:
            return 'Positive'

    #Function to prepair data for training
    def train(pol):
        if pol<0:
            return 'Negative'
        elif pol != 0:
            return 'Positive'
        
    df["Score"]=df['Polarity'].apply(getTextAnalysis)
    df["Label"]=df['Polarity'].apply(assignLabels)
    df["Train"]=df['Polarity'].apply(train)
    print("\n\nPRINTING THE FINAL PROCESSED COLLECTION OF TWEETS : \n")
    print(df.head(10))
    print("\n\nTHE VECTOR TABLE : \n")
    twitter_countvectorizer.shape
    
    print("\n\nSHOWING THE PERCENTAGE OF POSITIVE, NEAGTIVE AND NEUTRAL TWEETS OF THE ENTERED TWITTER HANDLE : \n")
    #Calculating Percentage of Positive Tweets
    positive= df[df['Polarity']>0]
    print(str(positive.shape[0]/(df.shape[0])*100)+"% of tweets are Positive.")
    pos=positive.shape[0]/df.shape[0]*100

    #Calculating Percentage of Neagtive Tweets
    negative= df[df['Polarity']<0]
    print(str(negative.shape[0]/(df.shape[0])*100)+"% of tweets are Negative.")
    neg=negative.shape[0]/df.shape[0]*100

    #Calculating Percentage of Neutral Tweets
    neutral= df[df['Polarity']==0]
    print(str(neutral.shape[0]/(df.shape[0])*100)+"% of tweets are Neutral.")
    neutrall=neutral.shape[0]/df.shape[0]*100

    #Plotting a Pie Chart to show Positive, Negative and Neutral Tweet Percentage
    explode=(0,0.1,0)
    labels='Positive','Negative','Neutral'
    sizes=[pos,neg,neutrall]
    colors=['yellowgreen','lightcoral','gold']
    plt.pie(sizes,explode,colors=colors,autopct='%1.1f%%',startangle=90)
    plt.legend(labels,loc=(-0.05,0.05),shadow=True)
    plt.axis('equal')
    plt.title("Pie Chart For Tweet Percentages")
    plt.show()

    #Plotting a Bar graph to show the Positive, Negative and Neutral Tweet Percentage
    labels = 'Positive','Negative','Neutral'
    plt.title("Bar Graph For Tweet Percentages")
    plt.ylabel("Number Of Tweets")
    plt.bar(labels, sizes, color=colors)
    plt.show()
    
    #Plotting a Scatter graph between Subjectivity and Polarity
    for index, row in df.iterrows():
        if row['Polarity']>0:
            plt.scatter(row['Polarity'],row['Subjectivity'],color='green')
        elif row['Polarity']<0:
            plt.scatter(row['Polarity'],row['Subjectivity'],color='red')
        elif row['Polarity']==0:
            plt.scatter(row['Polarity'],row['Subjectivity'],color='blue')
    plt.title('Twitter Sentimental Analysis')
    plt.xlabel('Ploarity')
    plt.ylabel('Subjectivity')
    plt.show()    
    df2 = df.drop(df[df['Label']=='Neutral'].index)
    twitter_countvectorizer = vectorizer.fit_transform(df2['Tweet'])
    twitter = pd.DataFrame(twitter_countvectorizer.toarray())
    X = twitter
    print(X)
    y = df2['Score']
    print(X.shape)
    print(y.shape)
    #Training my model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #Applying the Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    NB_classifier = MultinomialNB()
    NB_classifier.fit(X_train, y_train)
    
    #Predicting the Test set results
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    y_predict_test = NB_classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_predict_test)
    
    print("\n\n\n PRINTING THE CLASSIFICATION REPORT USING NAIVE BAYES : \n\n\n")
    print(classification_report(y_test, y_predict_test))
    sns.heatmap(cm, annot=True)
    plt.xlabel('TRUE VALUES')
    plt.ylabel('PREDICTED VALUES')
    plt.title('Heat Map of Naive Bayes ')
    plt.show()
    
    #Using SVM Classifier on my Data Frame
    df3 = df.drop(df[df['Label']=='Neutral'].index)
    trainData=df3.sample(frac=0.8,random_state=200) #random state is a seed value
    testData=df3.drop(trainData.index)    
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create feature vectors
    vectorizer = TfidfVectorizer(norm = False, smooth_idf = False)
    train_vectors = vectorizer.fit_transform(trainData['Tweet'])
    test_vectors = vectorizer.transform(testData['Tweet'])
    import time
    from sklearn import svm
    from sklearn.metrics import classification_report

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, trainData['Train'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    #Printing results of SVM
    print("\n\nPRINTING CLASSIFICATION REPORT USING SVM: \n")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    report = classification_report(testData['Train'], prediction_linear, output_dict=True)
    print('Positive: ', report['Positive'])
    print('Negative: ', report['Negative'])
    print(classification_report(testData['Train'], prediction_linear))
    cm2 = confusion_matrix(testData['Train'], prediction_linear)
    #f1score= 2*((precision * recall)/( precision + recall))
    sns.heatmap(cm2, annot=True)
    plt.xlabel('TRUE VALUES')
    plt.ylabel('PREDICTED VALUES')
    plt.title('Heat Map of SupportVector Machine (SVM) ')
    plt.show()

#Creating the Window of the GUI
root.geometry("633x333")
root.maxsize(1200, 800)
root.minsize(400, 200)

#Personalizing the GUI
root.title("Social Media Sentimental Analysis by Dhruva Bisht")#adding a title to my title window of the gui
title_On_Screen = Label(text = "ENTER THE TWITTER HANDEL BELOW", fg = "blue" , font = ("bold", 22), borderwidth = 10, relief = GROOVE)
title_On_Screen.pack(fill = X)

#Getting the data from the user by taking an Input
twitterAccount_Entry = StringVar()
twitterAccount = Entry(root, textvariable = twitterAccount_Entry, borderwidth = 7, relief = GROOVE)
twitterAccount.pack()

#Now Creating the buttons to make the GUI interrative and attractive
button = Button(root, fg = "Black",  text = "SEARCH", font = ("Bold", 18), borderwidth = 7, relief = GROOVE,command = Twitter)
button.pack()#In the  button we have created a command to input the twitter handel

root.mainloop()


