# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (Univariate)
x = np.array([1, 2, 3, 4, 5])     # Input feature
y = np.array([2, 4, 5, 4, 5])     # Target values

# Number of observations
n = len(x)

# Calculate slope (m) and intercept (c)
m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
c = (np.sum(y) - m * np.sum(x)) / n

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Predict y values
y_pred = m * x + c

# Plot the data points and regression line
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Univariate Linear Regression using Least Squares')
plt.legend()
plt.show()

## Output:
<img width="1920" height="1080" alt="Screenshot (85)" src="https://github.com/user-attachments/assets/458453da-d9e0-4d5a-9ade-47821fe4a59b" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
