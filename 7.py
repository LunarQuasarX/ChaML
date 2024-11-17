import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error

X_class, y_class = make_classification(
    n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42
)

X_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

param_grid_svc = {"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01], "kernel": ["rbf"]}
grid_svc = GridSearchCV(SVC(), param_grid_svc, refit=True, verbose=2)
grid_svc.fit(X_train_class, y_train_class)
best_params_svc = grid_svc.best_params_
print(f"Best parameters for SVC: {best_params_svc}")

param_grid_svr = {"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01], "kernel": ["rbf"]}
grid_svr = GridSearchCV(SVR(), param_grid_svr, refit=True, verbose=2)
grid_svr.fit(X_train_reg, y_train_reg)
best_params_svr = grid_svr.best_params_
print(f"Best parameters for SVR: {best_params_svr}")

y_pred_class = grid_svc.predict(X_test_class)
print(f"Classification Accuracy: {accuracy_score(y_test_class, y_pred_class)}")

y_pred_reg = grid_svr.predict(X_test_reg)
print(f"Regression Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_reg)}")

xx, yy = np.meshgrid(
    np.linspace(X_class[:, 0].min() - 1, X_class[:, 0].max() + 1, 100),
    np.linspace(X_class[:, 1].min() - 1, X_class[:, 1].max() + 1, 100),
)
Z = grid_svc.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
plt.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap="viridis")
plt.title("SVC Decision Boundary (2D)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_class[:, 0], X_class[:, 1], y_class, c=y_class, cmap="viridis")
ax.set_title("SVC Data Points (3D)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Class Label")
plt.show()

xx, yy = np.meshgrid(
    np.linspace(X_reg[:, 0].min(), X_reg[:, 0].max(), 100),
    np.linspace(X_reg[:, 1].min(), X_reg[:, 1].max(), 100),
)
Z = grid_svr.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_reg[:, 0], X_reg[:, 1], c=y_reg, cmap="coolwarm", edgecolor="k")
plt.title("SVR Regression Predictions (2D)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_reg[:, 0], X_reg[:, 1], y_reg, c=y_reg, cmap="coolwarm", marker="o")
ax.set_title("SVR Data Points and Regression Surface (3D)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Target")
plt.show()
