from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_decision_tree(criterion, min_samples_split, min_samples_leaf, max_depth):
    clf = DecisionTreeClassifier(
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Criterion: {criterion}, Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}, Max Depth: {max_depth}")
    print(f"Accuracy: {accuracy}\n")

evaluate_decision_tree(criterion='entropy', min_samples_split=2, min_samples_leaf=1, max_depth=None)  # ID3
evaluate_decision_tree(criterion='gini', min_samples_split=2, min_samples_leaf=1, max_depth=None)     # C4.5

evaluate_decision_tree(criterion='entropy', min_samples_split=4, min_samples_leaf=2, max_depth=3)
evaluate_decision_tree(criterion='gini', min_samples_split=4, min_samples_leaf=2, max_depth=3)