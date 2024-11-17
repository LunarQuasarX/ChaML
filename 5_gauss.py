from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gaussian_nb = GaussianNB()
print(X_train)
print(y_train)
gaussian_nb.fit(X_train, y_train)
y_pred_gaussian = gaussian_nb.predict(X_test)
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian)
print(f"Accuracy of GaussianNB: {accuracy_gaussian}")

