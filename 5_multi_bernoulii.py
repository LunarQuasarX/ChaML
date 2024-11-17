from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score

newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_vectorized, y_train)
y_pred_mnb = mnb.predict(X_test_vectorized)

print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_mnb))
print(classification_report(y_test, y_pred_mnb, target_names=newsgroups.target_names))

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_vectorized, y_train)
y_pred_bnb = bnb.predict(X_test_vectorized)

print("\nBernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_bnb))
print(classification_report(y_test, y_pred_bnb, target_names=newsgroups.target_names))
