#Step 1: Import all libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

#Step 2: Load the data
d = datasets.load_iris() # famous built in flower data set from sklearn

#Explore the data
print(list(d.keys()))

#print(d['DESCR'])

#Step 4: Split data
x = d['data'][:,3:]
print(x)

y = (d['target']==2).astype(np.int)

#Step 5: Create ml
model = LogisticRegression()

#Step 6: Train the machine model using fit function
model.fit(x,y)

#Step 7: Predict the machine model
#prediction = model.predict(([1.6]))

#Step 8: Evaluate the machine model
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = model.predict_proba(x_new)
print(y_prob[:,1])
plt.plot(x_new, y_prob[:,1],'g-',label='verginca')
plt.show()

















