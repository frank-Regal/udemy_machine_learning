# Step 1: Import all libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2: Load and read the data
d = pd.read_csv('Books.csv')

# Input data (Remove the Books columns because this is the desired values)
x = d.drop(columns=['Books'])

# Output data (desired data)
y = d['Books']

# Step 4: Split the data in train test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Step 5: Create the machine model
model = DecisionTreeClassifier()

# Step 6: Train the machine model
model.fit(x_train,y_train)

# Step 7: Prediction of the machine model
prediction = model.predict(x_test)

# Step 8: Final Evaluation of the machine model
score = accuracy_score(y_test,prediction)
print(score)

prediction = model.predict(([25,1],[20,0]))
print(prediction)