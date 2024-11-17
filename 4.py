import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_diabetes()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = (y - y.mean()) / y.std()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def ridge_regression(X, y, learning_rate=0.01, epochs=1000, alpha=1.0):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for epoch in range(epochs):
        y_pred = X.dot(weights) + bias
        dw = (-2 / m) * (X.T.dot(y - y_pred)) + 2 * alpha * weights
        db = (-2 / m) * np.sum(y - y_pred)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

def lasso_regression(X, y, learning_rate=0.01, epochs=1000, alpha=1.0):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for epoch in range(epochs):
        y_pred = X.dot(weights) + bias
        dw = (-2 / m) * (X.T.dot(y - y_pred)) + alpha * np.sign(weights)
        db = (-2 / m) * np.sum(y - y_pred)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

ridge_weights, ridge_bias = ridge_regression(X_train, y_train, alpha=0.1)
lasso_weights, lasso_bias = lasso_regression(X_train, y_train, alpha=0.1)

y_pred_ridge = X_test.dot(ridge_weights) + ridge_bias
y_pred_lasso = X_test.dot(lasso_weights) + lasso_bias

mse_ridge = np.mean((y_test - y_pred_ridge) ** 2)
mse_lasso = np.mean((y_test - y_pred_lasso) ** 2)

print("Ridge Reg:", mse_ridge)
print("Lasso Reg:", mse_lasso)
