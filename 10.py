from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

data = load_iris()
X = data.data
y_true = data.target 

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

ari_score = adjusted_rand_score(y_true, y_kmeans)
print("KMeans Adjusted Rand Index Score:", ari_score)

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Backpropagation Neural Network Accuracy:", accuracy)



data = load_iris()
X = data.data
y_true = data.target 

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

ari_score = adjusted_rand_score(y_true, y_kmeans)
print("KMeans Adjusted Rand Index Score:", ari_score)
