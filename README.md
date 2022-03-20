## Twitter Sentiment Analysis
This is a repository for the real time twitter sentiment analysis done using python along with a Graphical User interface (GUI).
- Contains an engaging GUI 
- Real Time analysis over any twitter account of your choice
- Fast, efficient and trust worthy

## Features
Once set up and installed this can help you perfrom real time anaylsis over any of the twitter account and further help your business as well as your studies to learn how to successfully setp up and use please follow along.

*Why use twitter for performing the analysis?*

-  Easy to gain access to a developer's account required for performing analysis
-  Over 290.5 million monthly active users worldwide
-  Users are projected to keep increasing up to over 340 million users by 2024

## Tech

- [Python 3](https://www.python.org/)
- [VsCode](https://code.visualstudio.com/)
- [A twitter account](https://twitter.com/)
 
And that's all you need for this project 😃

## Installation
Install the dependencies and devDependencies for the project to work perfectly, if you already have these libraries installed you can skip this step but it is always a good practise to keep all the libraries updated.

```sh
pip3 install tweepy
pip3 install textblob
pip3 install numpy
pip3 install pandas
pip3 install nltk
pip3 install stopwords
```

## Description

*What is sentiment analysis?*

>  It is the process of computationally identifying and categorizing opinions
>  expressed in a piece of text, especially in order to determine whether the
>  writer's attitude towards a particular topic, product, etc. is positive,
>  negative, or neutral.

The process is similar to any other text classification process, the tweets are fist collected directly from twitter, thentext cleaning is perfromed where all the punctuations, @ symbols (called mentions), # tags and urls are removed along with the common stop words for the english language. These are then converted into vectors, as computers can only understand 0's and 1's, which has been achieved using tfidf vectorization. The vector values are then used to find the polarity and subjectivity of the tweets.
In this project the polarity and subjectivity of the tweets has been used as a means to distinguish between positive, negative and neutral tweets.

*What is polarity and subjectivity of a tweet?*

>  The polarity numeric number indicates how negative or positive a sentence is.
>  Subjectivity, on the other hand, refers to how objective or subjective a text is.
>  TextBlob employs a sentiment-calculating algorithm, the “averaging” technique 
>  which is applied to polarity values to calculate a polarity score for a single word
>  and thus a similar procedure applies to every single word, resulting in a combined
>  polarity for larger texts.

*Why are we using tweepy?*

Tweepy is an open source Python package that gives you a very convenient way to access the Twitter API with Python. Tweepy includes a set of classes and methods that represent Twitter's models and API endpoints, and it transparently handles various implementation details, such as: Data encoding and decoding.

Then we have made a few graphs and charts to represent the percentage of positive, negative and neutral tweets to the user, for easy undersatnding and analysis. The dataset is now ready to be sent to classifiers for classification and prediction, the two classifiers used here are the naive bayes classifier and the support vector machines (SVM). there classification reports are then displayed to showcase the accuracy of the models.


## Applications of Sentiment Analysis

- **Analyzing Movie Reviews** For analyzing online movie reviews to gather audience insights into the film.
- **News sentiment Analysis** The technique of analyzing news sentiments for a certain company to gain insights. Examine the emotions expressed in theTwitter tweets.
- **Online food reviews** To analyzing user comments to discover how people feel about food.
- **E-Commerce and Social Networking** Users can submit text reviews, comments, or feedback on things on many social networking platforms or e-commerce websites.These user-generated texts are a great source of user sentiment opinions on a wide range of products and items. For an item, such language might potentially expose both the connected aspects of the item as well as the users’ opinions on each feature.



## Contributors
- Dhruva bisht <Dhruvabisht_19021048.cse@geu.ac.in>

## License & copyright
© Dhruva Bisht 2022
Licensed under the [GNU General Public License](LICENSE).

Happy Analysis ✌
