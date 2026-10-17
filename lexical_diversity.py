# MEMBER 1 - Length & vocabulary.  Input: ifnd_text.csv -> Output: features_m1.csv
# Core:  word_count, ttr, avg_word_len
# Extra: hapax_ratio, long_word_share, digit_count, starts_with_question_word
import re
import numpy as np
from common import load_clean, tokenize


def word_count(text):
# Number of word tokens
    return len(tokenize(text))

def ttr(text):
# Type-Token Ratio = unique words / total words.
    w = tokenize(text)
    return len(set(w)) / max(len(w), 1)

def avg_word_len(text):
# Mean characters per word
    w = tokenize(text)
    return float(np.mean([len(x) for x in w])) if w else 0.0


def hapax_ratio(text):
#Share of words that appear exactly once in the headline
    w = tokenize(text)
    return sum(w.count(x) == 1 for x in w) / max(len(w), 1)

def long_word_share(text):
# Share of words with 9+ letters
    w = tokenize(text)
    return sum(len(x) >= 9 for x in w) / max(len(w), 1)

def digit_count(text):
    # Number of digit characters (statistics / dates).
    return sum(c.isdigit() for c in text)

def starts_with_question_word(text):
# 1 if the headline starts with who/what/why/how/is/does/... else 0
    return int(bool(re.match(r"\s*(who|what|why|how|is|are|does|did|can|will|has|have)\b", text, re.I)))

FEATURES = [word_count, ttr, avg_word_len, hapax_ratio, long_word_share, digit_count, starts_with_question_word]

if __name__ == "__main__":
    df = load_clean()
    out = df[["row_id"]].copy()
    for f in FEATURES:
        out[f.__name__] = df["text"].apply(f)
    out.to_csv("features_m1.csv", index=False)
    print(out.describe().round(3).T)
