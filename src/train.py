import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import preprocess


def load_data(path):
    return pd.read_csv(path, sep="\t", names=["label", "message"])


def train_models(X_train, y_train):
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=1000)

    nb.fit(X_train_tfidf, y_train)
    lr.fit(X_train_tfidf, y_train)

    return vectorizer, nb, lr


def evaluate(model, X_test_tfidf, y_test, name):
    y_pred = model.predict(X_test_tfidf)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def main():
    df = load_data("../data/SMSSpamCollection")
    df = preprocess(df)

    X = df["clean_message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    vectorizer, nb, lr = train_models(X_train, y_train)
    X_test_tfidf = vectorizer.transform(X_test)

    evaluate(nb, X_test_tfidf, y_test, "Naive Bayes")
    evaluate(lr, X_test_tfidf, y_test, "Logistic Regression")

    joblib.dump(nb, "../models/spam_model.pkl")
    joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()

