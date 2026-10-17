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
        
    valid_years = ["18", "19", "20", "2018", "2019", "2020"]
    
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
  if row["Statement"][0] == '\"':
    return row["Statement"][1:-1]
  else:
    return row["Statement"]

def drop_non_utf8(row):
    text = row["Statement"]
    if pd.isna(text):
        return text
        
    return str(text).encode("utf-8", errors="ignore").decode("utf-8")

df = pd.read_csv("IFND.csv")
df["Statement"] = df.apply(fix_quotes, axis=1)
df["Date"] = df.apply(standardize_date, axis=1)
df["label_binary"] = df["Label"].map({"TRUE": 0, "Fake": 1}) 
df = df.drop_duplicates("Statement")
df["Statement"] = df.apply(drop_non_utf8, axis=1)

print(df["Date"].unique())



 
    