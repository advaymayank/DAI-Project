#here we track the following parameters:
#percentage capatilization in heading 
#letter frequency 
#vowel to consant ratio
#number of words that are all caps
#percentage of words that start with a capital letter
#number of quotation marks
import re
from common import load_clean
def upper_ratio(text):
    return sum(c.isupper() for c in text) / max(len(text), 1) #added fallback in case of zero length
def letter_freq(text,c):
    c='e' 
    return text.lower().count(c) / max(len(text.replace(" ", "")), 1) #added fallback in case of zero length
def vowel_ratio(text):
    return sum(c in "aeiou" for c in text.lower()) / max(len(text), 1) #added fallback in case of zero length
def all_caps_words(text):
    return sum(w.isupper() and len(w) >= 2 for w in re.findall(r"[A-Za-z]+", text)) #dont count single letter words
def title_case_ratio(text):
    w = re.findall(r"[A-Za-z]+", text)
    return sum(x[0].isupper() for x in w) / max(len(w), 1)
def quote_count(text):
    return sum(text.count(q) for q in ['"', "'"])
FEATURES = [upper_ratio, letter_freq, vowel_ratio, all_caps_words, title_case_ratio, quote_count]
if __name__ == "__main__":
    df = load_clean()
    out = df[["row_id"]].copy()
    for f in FEATURES:
        out[f.__name__] = df["text"].apply(f)
    out.to_csv("features_m3.csv", index=False)
    print(out.describe().round(3).T)
