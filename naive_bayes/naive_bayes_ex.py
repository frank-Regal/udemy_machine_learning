# Step 1: Import all libraries

import numpy as np

# Take datasets submodules - fetch_20newsgroups is a famous group is a tokenization data
from sklearn.datasets import fetch_20newsgroups

# This identifies the weight of the words (physically weights the words)
from sklearn.feature_extraction.text import TfidfVectorizer

# import mutinominal naive bayes which is the classifier 
from sklearn.naive_bayes import MultinomialNB

# this passes the weight of the words to the multinominal naive bayes
from sklearn.pipeline import make_pipeline

# Analysis of the classifier
from sklearn.metrics import confusion_matrix

# Step 2: Load the data
data = fetch_20newsgroups() # built in data that exists in sklean library

# Explore the data
categories = data.target_names

# Step 3: Clean the data, but the data is already clean

# Step 4: Split the data in train and test
train_data = fetch_20newsgroups(subset='train',categories=categories) # already allocated
test_data = fetch_20newsgroups(subset='test',categories=categories)   # already allocated

# Step 5: Create the model based on Multinomial Naive bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Step 6: Train the model
model.fit(train_data.data,train_data.target)

# Step 7: Prediction 
prediction = model.predict(test_data.data)

# Step 8: Final Evaluation
def pred_category(s, train=train_data, model=model):
    prediction = model.predict([s])
    return train.target_names[prediction[0]]

print('Categories:')
str1 = 'Jesus Christ'
print("String 1: " + str1)
print(pred_category(str1))

print(pred_category('International space station'))

print(pred_category('lamborgini is better than ferrari'))

print(pred_category('President of America'))

print(pred_category('flyers'))
print(" ")

print("Where does sex belong?: ")
print(pred_category('the doctor found out the childs sex'))

