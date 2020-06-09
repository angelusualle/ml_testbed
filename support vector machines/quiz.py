# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# Read the data.
data = pd.DataFrame(np.asarray(pd.read_csv('data.csv', header=None)))
# Assign the features to the variable X, and the labels to the variable y. 
X = data.iloc[:,0:2]
y = data.iloc[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = GridSearchCV(estimator=SVC( kernel='rbf'), param_grid={
    'gamma': range(0,30)
})

model.fit(X , y)

# TODO: Fit the model.

print(model.best_params_)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
print(acc)