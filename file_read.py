import re   # added for text repair
import pandas as pd

def standardize_date(row):
    val = row["Date"]
    
    if pd.isna(val):
        return pd.NA
        
    val_str = str(val).strip()
    if val_str.lower() in ["nan", "na", "none", ""]:
        return pd.NA

    parts = val_str.split("-")
    if len(parts) < 2:
        return pd.NA
        
    valid_years = [str(y) for y in range(2011, 2022)] + [str(y)[-2:] for y in range(2011, 2022)]
    
    if parts[-1] in valid_years:
        year = parts[-1][-2:] 
        month = parts[-2]    
    elif parts[0] in valid_years:
        year = parts[0][-2:]
        month = parts[1]
    else:
        return pd.NA
        
    try:
        reconstructed = f"{month}-{year}"
        
        dt = pd.to_datetime(reconstructed, format="%m-%y", errors="coerce")
        
        if pd.isna(dt):
            dt = pd.to_datetime(reconstructed, format="%b-%y", errors="coerce")
            
        if not pd.isna(dt):
            return dt.strftime("%m-%y") 
            
        return pd.NA
        
    except Exception:
        return pd.NA
  

def fix_quotes(row):
  if len(row["Statement"]) > 1 and row["Statement"][0] == '\"' and row["Statement"][-1] == '\"':
    return row["Statement"][1:-1]
  else:
    return row["Statement"]

SYMBOLS = {"Õ": "'", "Ô": "'", "Ò": '"', "Ó": '"', "Ð": "-", "Ñ": "-", "É": "...",
           "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x96": "-", "\x97": "-", "\x85": "...",
           "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}

def drop_non_utf8(row):
    text = row["Statement"]
    if pd.isna(text):
        return text
    text = str(text)
    if re.search(r"[âÃ][\x80-\xbf]", text):
        try: text = text.encode("latin1").decode("utf-8")
        except UnicodeError: pass
    for bad, good in SYMBOLS.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()

df = pd.read_csv("IFND.csv", encoding="latin1")      
df = df.dropna(subset=["Statement"])                 
df["Statement"] = df.apply(drop_non_utf8, axis=1)    
df["Statement"] = df.apply(fix_quotes, axis=1)
df["Date"] = df.apply(standardize_date, axis=1)
df["label_binary"] = df["Label"].map({"TRUE": 0, "Fake": 1}) 

df = df[df.groupby("Statement")["label_binary"].transform("nunique") == 1]
df = df.drop_duplicates("Statement")


df["Category"] = df["Category"].replace({"MISLEADIND": "MISLEADING"})
df["year"] = pd.to_numeric("20" + df["Date"].str[-2:], errors="coerce").astype("Int64")
df["month"] = pd.to_numeric(df["Date"].str[:2], errors="coerce").astype("Int64")
df["year_month"] = df["year"].astype("string") + "-" + df["month"].astype("string").str.zfill(2)
df["date_quality"] = df["Date"].notna().map({True: "parsed", False: "missing"})
df = df.rename(columns={"Statement": "text"}).reset_index(drop=True)
df.insert(0, "row_id", df.index)
df[["row_id", "text", "label_binary", "Web", "Category",
    "year", "month", "year_month", "date_quality"]].to_csv("ifnd_clean.csv", index=False)

print(len(df), "rows saved to ifnd_clean.csv")
print(df["Date"].unique())
