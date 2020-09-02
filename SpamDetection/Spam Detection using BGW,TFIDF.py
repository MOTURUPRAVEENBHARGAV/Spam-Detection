# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:58:52 2020

@author: ganeshmaruti
"""

### SPAM DETECTION PROJECT USING BAG OF WORDS AND TFIFDF ALGORITHMS

#importing necessary Libraries

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




messages=pd.read_csv("H:/DSE/NLP/Krish Naik ( NLP )/SpamDetection/SMSSpamCollection",sep='\t',names=['label','messages'])
messages.head()

#Data Cleaning and preprocessing

corpus=[]
lemmatizer = WordNetLemmatizer()


for i in range(len(messages)):
    review=re.sub(r'[^a-zA-z]'," ",messages.messages[i])
    review = review.lower()
    review= review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=" ".join(review)
    corpus.append(review)
    
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()  

lb=LabelEncoder()
y=lb.fit_transform(messages.label)
#y=pd.get_dummies(messages['label'],drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

#Training model using Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train, y_train)

y_pred= spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_mtx=confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test,y_pred)



# USING Tfidf MODEL

tf=TfidfVectorizer(max_features=5000)
X_tf=tf.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_tf, y, test_size=0.30, random_state=1)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model_tf=MultinomialNB().fit(X_train, y_train)

y_pred_tf= spam_detect_model.predict(X_test_tf)

from sklearn.metrics import confusion_matrix

confusion_mtx_tf=confusion_matrix(y_test_tf,y_pred_tf)

accuracy_tf=accuracy_score(y_test_tf,y_pred_tf)


