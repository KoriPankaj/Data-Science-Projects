############# DATA MINING ############
##### SUPERVISED LEARNING ######
#### NAIVE BAYES ####
# Q-3 DISASTER TWEETS DATASET

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Loading the data set
twt_data = pd.read_csv(r"E:\360digitMG\ASsignments\Naive bayes\Disaster_tweets_NB.csv", encoding = "ISO-8859-1")

# cleaning data 

# ---------
stop_words = []
# Load the custom built Stopwords
with open(r"E:\360digitMG\Text mining & NLP\Datasets NLP\stop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
# ---------

import re

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+", " ", i).lower()
#    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word) > 3:
            w.append(word)
    return (" ".join(w))

# testing above function with sample text => removes punctuations, numbers
cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi hw .. are you?")
cleaning_text("Hi how are you, I am sad")

twt_data.text = twt_data.text.apply(cleaning_text)

# removing empty rows
twt_data = twt_data.loc[twt_data.text != "", :]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

twt_train, twt_test = train_test_split(twt_data, test_size = 0.2)


# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
twt_bow = CountVectorizer(analyzer = split_into_words).fit(twt_data.text)

# Defining BOW for all messages
all_twt_matrix= twt_bow.transform(twt_data.text)

# For training messages
train_twt_matrix = twt_bow.transform(twt_train.text)

# For testing messages
test_twt_matrix = twt_bow.transform(twt_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_twt_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_twt_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_twt_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, twt_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)

pd.crosstab(test_pred_m, twt_test.target)

accuracy_test_m = np.mean(test_pred_m == twt_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, twt_test.target) 


# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == twt_train.target)
accuracy_train_m

pd.crosstab(train_pred_m, twt_train.target)

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

#formula: 
# P(w|spam) = (num of spam with w + alpha)/(Total num of spam emails + K(alpha))
# K = total num of words in the email to be classified

classifier_mb_lap = MB(alpha = 0.8)
classifier_mb_lap.fit(train_tfidf, twt_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == twt_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, twt_test.target) 

pd.crosstab(test_pred_lap, twt_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == twt_train.target)
accuracy_train_lap

pd.crosstab(train_pred_lap, twt_train.target)

''' 91.2 % accuracy for train data, where as 79% accuracy for test data '''