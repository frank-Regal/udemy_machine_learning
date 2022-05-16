
# // Step 1: Load the Libraries 
# import standard python libraries
import pandas as pd   # "python data analysis"  (used to read csv files)
import numpy as np    # used to create arrays in python language

# sklearn is a famous library in python for machine learning
from sklearn.model_selection import train_test_split  # this is used to split the data into train and test sets
from sklearn.neighbors import KNeighborsClassifier    # this is the classifier for KNN
from sklearn.metrics import classification_report, confusion_matrix # reporting
from sklearn.preprocessing import StandardScaler      # standard scaler is used to convert units to an equal to all the data

# // Step 2: Load and Read the Data with pandas
d = pd.read_csv('Data.csv',index_col=0) # index_col=0 says take all the rows of the columns, index=1 says to take all the columns of the row

# // Step 2.5: Explore and Standardize the data
d.head()

# Standardize
scaler = StandardScaler()

# fit function for training of the model. Fitting only the input data without the output class
scaler.fit(d.drop('TARGET CLASS', axis =1)) # target class is the output of the data, droping axis is 1 because you want to drop the column

# transform the data. Do a proper transformation to transform the data to 1 standard scale. This must be done when scaling data
scaled = scaler.transform(d.drop('TARGET CLASS', axis =1))

# take independent features of data
# DataFrame 
ind = pd.DataFrame(scaled, columns=d.columns[:-1]) # do not include the last column from the data

# // Step 3: Clean the data (Data is already clean)

# // Step 4: Split the data into the train and test
x_train, x_test, y_train, y_test = train_test_split(scaled,d['TARGET CLASS'],test_size=0.30) 

# // Step 5: Now create the machine model
model = KNeighborsClassifier(n_neighbors=23) # one line of code for a machine model

# // Step 6: Train the model using fit method
model.fit(x_train,y_train)

# // Step 7: Prediction of the model
prediction = model.predict(x_test)

# // Step 8: Evaluation of the model
# Check confusion matrix
print(confusion_matrix(y_test,prediction))

# Check the classification report
print(classification_report(y_test,prediction))