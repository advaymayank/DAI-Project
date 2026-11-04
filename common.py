import re
import pandas as pd

def load_clean(filepath="ifnd_clean.csv"):
    return pd.read_csv(filepath)

def tokenize(text):
    if pd.isna(text):
        return []
    # Find all alphabetic word tokens and convert to lowercase
    return re.findall(r"\b[A-Za-z]+\b", str(text).lower())