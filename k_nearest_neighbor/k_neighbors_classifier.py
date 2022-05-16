# -*- coding: utf-8 -*-
"""K_neighbors_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1npvZ6hHaa4Bqw059lHiCoVcbQGkoGiBV

**K-Neighbors Classifier**
"""

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::: KNN Classifier ::::::::::::::::::::::::::::::::::::::::::::::

#Step1:
#Import all the libraries
import pandas as pd
import numpy as np  #It creates multi dimensional array
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

#Step 2:
#Load and read the dataset
d = pd.read_csv('Data.csv',index_col = 0)
#Set index_col = 0, it will automatically set these feature names in first row.
#Target class has 1s and 0s all the way

d.head()

#Standardize the data
''' The respective dataset may be calculated in some different units or weights, so we need to bring it back to 
the normal scale. Therefore we need its standardization where we have all the data in similar form.To do so we are gonna use 
standard scaler for scalling. For standard scaler we are gonna import the StandardScaler from sklearn.preprocessing.
 '''

#Now to create an object of standard scaler
scaler  = StandardScaler()

#Now to do the fit, where we are gonna drop the TARGET CLASS, because it a dependent feature 
# We need standard scalling on independent features(inputs with weights), not dependent. so drop the dependent feature(TARGET CLASS)

scaler.fit(d.drop('TARGET CLASS', axis = 1))

#Now to do the transformation
scaled = scaler.transform(d.drop('TARGET CLASS',axis = 1)) #We did transformation of the data and stored it in scaled
#axis = 1, taking all the columns execpt TARGET CLASS.

#The next step is to take the independent features
ind = pd.DataFrame(scaled,columns = d.columns[:-1]) # skipping the last column and picking all the other columns
ind.head()

#Step 3: Cleaning of the data. Data is already clean, so we need to skip step 3 here

#Step 4
#Now to do the training and testing procedure
x_train, x_test, y_train, y_test = train_test_split(scaled,d['TARGET CLASS'],test_size = 0.30)
# scaled are independent features, 'TARGET CLASS' is dependent feature and the test size is 30%

# Step 5:
#Now to create the KNN classifier with k = 1
model = KNeighborsClassifier(n_neighbors=1)

# Step 6:
#Now to train the model
model.fit(x_train, y_train)

# Step 7:
#Now to do the prediction
prediction = model.predict(x_test)

# Step 8:
# Final evaluation of the model
''' Confusion matrix tells us how good is the model. What's the precision and accuracey rate
 '''
print(confusion_matrix(y_test,prediction))

#Check the precision score
print(classification_report(y_test,prediction))  #still we have a better precision in below table, but how to get the exact k value? let's do it

#Now to increase the value of k, let's suppose my k=23
model = KNeighborsClassifier(n_neighbors= 23)
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))