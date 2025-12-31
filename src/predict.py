import joblib
from preprocess import clean_text

model = joblib.load("../models/spam_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

def predict_spam(text):
    vector = vectorizer.transform([clean_text(text)])
    return "Spam" if model.predict(vector)[0] == 1 else "Ham"


if __name__ == "__main__":
    msg = input("Enter message: ")
    print("Prediction:", predict_spam(msg))

