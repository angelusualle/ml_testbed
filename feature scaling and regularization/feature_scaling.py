# TODO: Add import statements
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header=None)
print(train_data.head())
X = train_data.iloc[:, :6]
y = train_data.iloc[:, 6]
print(X.head())
print(y.head())

# TODO: Create the standardization scaling object.
scaler = StandardScaler().fit(X)

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X_scaled, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)