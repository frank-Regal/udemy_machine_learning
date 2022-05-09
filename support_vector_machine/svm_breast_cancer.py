## Step 1: Load all the libraries
# sklearn is used for ml in python
from sklearn import datasets # load the datasets from sklearn
from sklearn.model_selection import train_test_split # library (used to split data into train and test)
from sklearn import svm # support vector machine
from sklearn import metrics # confusion matrix and model

## Step 2: Load the data
d = datasets.load_breast_cancer() 

# Now explore the data
#print(d)           # prints all the data in dataset
#print(d['data'])  # prints the target data from the dataset loaded to 'd' (binary form)


## Step 3: Clean the data (This is already done with the sklearn datasets)

## Step 4: Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(d.data,d.target,test_size=0.4,random_state=209)
# test_size = total amount of data used to test
# random_state = the random samples of the data

## Step 5: Create the machine model
model = svm.SVC(kernel='linear')
# three types of kernels:
    # 1. linear
    # 2. polynomial
    # 3. radial basis kernel
# A kernel is being used because the data is in a non-linear form, so therefore we are adding one dimension to the dataset with a linear kernel
# SVC: support vector classify
# SVR: support vector regressor

## Step 6: Train the machine model using fit method
# train the machine model
model.fit(x_train,y_train) 

## Step 7: Prediction of the model
prediction = model.predict(x_test)
# print(prediction)

## Step 8: Evaluation of the model
# precision, accuracy, confusion matrix
accuracy = metrics.accuracy_score(y_test,y_pred=prediction)
print('Accuracy: ', accuracy)
# Print classification report
c_report = metrics.classification_report(y_test,y_pred=prediction)
print(c_report)

