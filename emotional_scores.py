
# pip install vaderSentiment textblob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

_vader = SentimentIntensityAnalyzer()


def vader(text):
# VADER compound polarity, -1 (negative) to +1 (positive)
    return _vader.polarity_scores(text)["compound"]

def subjectivity(text):
# TextBlob subjectivity, 0 (objective) to 1 (subjective)
    return TextBlob(text).sentiment.subjectivity

def punct(text):
# Count of '!' + '?' + '...'
    return text.count("!") + text.count("?") + text.count("...")


CLICKBAIT = {"shocking", "breaking", "viral", "watch", "must", "amazing", "horrifying",
             "unbelievable", "exposed", "secret", "truth", "revealed", "urgent", "alert", "boycott"}
NEGATIONS = {"not", "no", "never", "nothing", "none", "neither", "nor", "cannot",
             "isn't", "wasn't", "don't", "didn't"}
DIRECT = {"you", "your", "we", "our", "us", "this", "these", "they", "them"}

def clickbait_words(text):
# Count of clickbait trigger words
    return sum(w in CLICKBAIT for w in tokenize(text))

def negation_count(text):
# Count of negation words (no, not, never...)
    return sum(w in NEGATIONS for w in tokenize(text))

def direct_address_count(text):
# Count of you / we / this / they style words
    return sum(w in DIRECT for w in tokenize(text))

def colon_or_dash(text):
# 1 if the headline has a 'label: text' or 'a - b' structure
    return int(":" in text or " - " in text)

FEATURES = [vader, subjectivity, punct, clickbait_words, negation_count, direct_address_count, colon_or_dash]

if __name__ == "__main__":
    df = load_clean()
    out = df[["row_id"]].copy()
    for f in FEATURES:
        out[f.__name__] = df["text"].apply(f)
    out.to_csv("features_m2.csv", index=False)
    print(out.describe().round(3).T)
