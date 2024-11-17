from sklearn.datasets import load_diabetes
import numpy as np

data = load_diabetes()
X = data.data[:, 2]
y = data.target
print(X)
print(y)
learning_rate = 0.01
iterations = 1000
m, b = 0, 0

for _ in range(iterations):
    y_pred = m * X + b
    dm = (-2 / len(X)) * np.sum(X * (y - y_pred))
    db = (-2 / len(X)) * np.sum(y - y_pred)
    m -= learning_rate * dm
    b -= learning_rate * db

print("Simple Linear Regression (Gradient Descent) - Slope:", m, "Intercept:", b)

X_mean = np.mean(X)
y_mean = np.mean(y)
m = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
b = y_mean - m * X_mean

print("Simple Linear Regression (Normal Equation) - Slope:", m, "Intercept:", b)

X = data.data
X_b = np.c_[np.ones((X.shape[0], 1)), X]
y = data.target
theta = np.zeros(X_b.shape[1])

for _ in range(iterations):
    y_pred = X_b.dot(theta)
    gradient = (2 / len(y)) * X_b.T.dot(y_pred - y)
    theta -= learning_rate * gradient

print("Multiple Linear Regression (Gradient Descent) - Weights:", theta)

theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Multiple Linear Regression (Normal Equation) - Weights:", theta)
