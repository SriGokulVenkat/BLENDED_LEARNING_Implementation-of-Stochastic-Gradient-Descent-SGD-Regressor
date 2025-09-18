# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries.


2.Load the dataset.


3.Preprocess the data (handle missing values, encode categorical variables).


4.Split the data into features (X) and target (y).


5.Divide the data into training and testing sets. 


6.Create an SGD Regressor model. 


7.Fit the model on the training data. 


8.Evaluate the model performance. 


9.Make predictions and visualize the results.
## Program:
```
Program to implement SGD Regressor for linear regression.
Developed by: SRI GOKUL VENKAT M
RegisterNumber: 212224040324
```
``` py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("CarPrice_Assignment .csv")
print(data.head())
print(data.info())

data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

X = data.drop('price', axis=1)
y = data['price']

scaler = StandardScaler() 
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

sgd_model.fit(X_train, y_train)

y_pred = sgd_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*50)
print("Name: SRI GOKUL VENKAT M")
print("Reg. No: 212224040324")
print("\nMean Squared Error:", mse)
print("R-squared Score:", r2)
print("="*50)

print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()
```
## Output:
<img width="530" height="292" alt="Screenshot 2025-09-18 154910" src="https://github.com/user-attachments/assets/253991cd-1e0c-40f4-9a98-96ed02d85055" />

<img width="475" height="382" alt="Screenshot 2025-09-18 154929" src="https://github.com/user-attachments/assets/c6b62825-9bcf-43e2-9aa1-9e17119a4708" />
<img width="1157" height="498" alt="Screenshot 2025-09-18 154941" src="https://github.com/user-attachments/assets/670a7208-29f6-42a4-af7c-ac52d7f39562" />
