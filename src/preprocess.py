import re
import string
import pandas as pd

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def preprocess(df: pd.DataFrame):
    df["clean_message"] = df["message"].apply(clean_text)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df

