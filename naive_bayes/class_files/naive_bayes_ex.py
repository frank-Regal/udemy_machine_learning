# -*- coding: utf-8 -*-
"""Naive_Bayes_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14sCQkIO3zWYeiR4XpH1Upreeoc2U0PXs

**NAIVE BAYES CLASSIFIER**
"""

# Step 1:
#Load all the libraries
import numpy as np  
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

# Step 2:
# Load the data
d = fetch_20newsgroups()  # we set data to a variable named 'd'

d.target_names

''' The above are catagories they have already assigned to this news group, these are called fetch_20 because there are 20 different topicss
or twenty different catagories here
 '''

# Now to define all the catagories and set up all the data.
# First to define all the catagories
categories = ['alt.atheism',  
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

# Step 3:
# Clean the data. Data has been already cleaned

# Step 4:
# Split the train and test data
train = fetch_20newsgroups(subset = 'train',categories=categories)

#Now to do the testing phase of the data
test = fetch_20newsgroups(subset='test',categories=categories)

print(len(train.d)) # It shows total no. of articles in train data which are 11314
print(len(test.d))   # It shows total no. of articles in test data which are 7532

# Step 5:
# Create the model
model = make_pipeline(TfidfVectorizer(),MultinomialNB())

# Step 6
# Train the model
model.fit(train.data, train.target)

# Step 7
# Prediction of the model
prediction = model.predict(test.d)
prediction

# Now to use that model and predicting category on new data based on trained model
def predict_category(s, train=train,model=model):
  prediction = model.predict([s])
  return train.target_names[prediction[0]]

# Now to do some predictions
predict_category('Jesus Christ')

predict_category('International space station')

predict_category('lamborghini is better than ferrari') #rec mean recreational

# Now if we put something like a caption
predict_category('President of America')  #it returns talk.politics miscellaneous