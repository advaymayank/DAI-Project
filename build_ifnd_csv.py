#!/usr/bin/env python3
"""
Build an analysis-ready CSV from the IFND (Indian Fake News Dataset).

IFND (Sharma & Garg, 2021) ships headlines + metadata + a link to the original
article, but NOT the article body. This script:
  1. loads IFND.csv (local path or the GitHub raw URL),
  2. normalises headline / date / real-fake label,
  3. downloads each article page and extracts the body text (cached + resumable),
  4. writes one CSV:

   index, headline, article_chars, article_words, real_fake_flag, label_name,
   datetime_published, article_text, text_source, url, scrape_status, category

Usage:
    pip install pandas requests beautifulsoup4 trafilatura   # trafilatura optional but better
    python build_ifnd_csv.py --input IFND.csv --output ifnd_text.csv
    python build_ifnd_csv.py --input IFND.csv --limit 50 --output sample.csv   # try first
    python build_ifnd_csv.py --input IFND.csv --no-scrape                       # headlines only

Re-running resumes from the cache file (--cache), so Ctrl-C is safe.
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

RAW_URL = "https://raw.githubusercontent.com/sonalgarg174/Dataset/main/IFND.csv"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 (academic research; IFND study)")

# candidate column names, matched case-insensitively
COLS = {
    "id":       ["id", "index", "idx"],
    "headline": ["statement", "news headings", "headline", "title"],
    "url":      ["web", "link", "link of news", "url", "news_link"],
    "date":     ["date", "date and time", "datetime", "published"],
    "label":    ["label", "class", "target"],
    "category": ["category", "topic"],
}


# --------------------------------------------------------------------------- load
def load_ifnd(src: str) -> pd.DataFrame:
    if re.match(r"https?://", src):
        raw = requests.get(src, headers={"User-Agent": UA}, timeout=60)
        raw.raise_for_status()
        df = pd.read_csv(StringIO(raw.content.decode("utf-8", errors="replace")))
    else:
        try:
            df = pd.read_csv(src, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(src, encoding="latin-1")
    return df


def pick(df: pd.DataFrame, key: str):
    lower = {c.strip().lower(): c for c in df.columns}
    for cand in COLS[key]:
        if cand in lower:
            return lower[cand]
    return None


def norm_label(v):
    """Return 1 for real/true, 0 for fake, None if unknown. Handles 1/0, TRUE/Fake, etc."""
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true", "real", "t"}:
        return 1
    if s in {"0", "0.0", "fake", "false", "f"}:
        return 0
    return None


# --------------------------------------------------------------------------- extract
class _TextParser(HTMLParser):
    """Stdlib fallback: collect <p> text, ignoring script/style/nav-ish tags."""
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form"}

    def __init__(self):
        super().__init__()
        self.skip = 0
        self.in_p = False
        self.paras, self.cur = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "p" and not self.skip:
            self.in_p = True

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "p" and self.in_p:
            t = " ".join("".join(self.cur).split())
            if t:
                self.paras.append(t)
            self.cur, self.in_p = [], False

    def handle_data(self, data):
        if self.in_p and not self.skip:
            self.cur.append(data)


def extract_text(html: str, url: str):
    """Return (text, published_date_or_None). trafilatura if installed, else fallbacks."""
    try:
        import trafilatura
        text = trafilatura.extract(html, url=url, include_comments=False,
                                   include_tables=False, favor_recall=True)
        meta = trafilatura.extract_metadata(html, default_url=url)
        pub = getattr(meta, "date", None) if meta else None
        if text:
            return text.strip(), pub
    except ImportError:
        pass
    except Exception:
        pass
    # fallback: BeautifulSoup if present, else stdlib parser
    pub = None
    m = re.search(r'(?:article:published_time|datePublished)["\']?\s*[:=]?\s*["\']?'
                  r'(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)', html)
    if m:
        pub = m.group(1)
    p = _TextParser()
    try:
        p.feed(html)
    except Exception:
        pass
    paras = [t for t in p.paras if len(t) > 40]  # drop menu/caption crumbs
    return "\n".join(paras).strip(), pub


# --------------------------------------------------------------------------- scrape
_host_lock = threading.Lock()
_host_last = {}


def _throttle(host: str, delay: float):
    """Polite per-host spacing so 8 threads don't hammer one site."""
    with _host_lock:
        now = time.monotonic()
        wait = _host_last.get(host, 0) + delay - now
        _host_last[host] = max(now, _host_last.get(host, 0) + delay)
    if wait > 0:
        time.sleep(wait)


def fetch_article(url: str, delay: float, timeout: int, retries: int = 2) -> dict:
    out = {"status": "no_url", "text": "", "pub": None}
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return out
    host = urlparse(url).netloc
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept-Language": "en-IN,en;q=0.9"})
    for attempt in range(retries + 1):
        try:
            _throttle(host, delay)
            r = sess.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                text, pub = extract_text(r.text, url)
                out.update(status="ok" if text else "empty_extract", text=text, pub=pub)
                return out
            out["status"] = f"http_{r.status_code}"
            if r.status_code in (403, 404, 410, 451):   # retrying will not help
                return out
        except requests.RequestException as e:
            out["status"] = f"error_{type(e).__name__}"
        time.sleep(1.5 * (attempt + 1))
    return out


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", default=RAW_URL, help="IFND.csv path or URL (default: GitHub raw)")
    ap.add_argument("--output", default="ifnd_text.csv")
    ap.add_argument("--cache", default="ifnd_scrape_cache.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delay", type=float, default=1.0, help="min seconds between requests to the same host")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--min-chars", type=int, default=200,
                    help="scraped text shorter than this is treated as a failed extraction")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N rows (for testing)")
    ap.add_argument("--dayfirst", action="store_true",
                    help="parse ambiguous dates like 03-06-2020 as 3 June (check your file's format)")
    ap.add_argument("--no-scrape", action="store_true", help="skip downloading; headlines only")
    args = ap.parse_args()

    df = load_ifnd(args.input)
    print(f"Loaded {len(df):,} rows; columns: {list(df.columns)}")
    c = {k: pick(df, k) for k in COLS}
    if c["headline"] is None:
        sys.exit("Could not find a headline column (looked for: %s)" % COLS["headline"])
    if c["url"] is None and not args.no_scrape:
        print("WARNING: no article-link column found; full text cannot be fetched. "
              "Falling back to headline-only (use --no-scrape to silence).")
        args.no_scrape = True
    if args.limit:
        df = df.head(args.limit)

    n = len(df)
    ids = df[c["id"]] if c["id"] else pd.Series(range(n), index=df.index)
    urls = df[c["url"]] if c["url"] else pd.Series([None] * n, index=df.index)

    # ---- scrape (resumable)
    cache_path = Path(args.cache)
    cache = {} if args.no_scrape else load_cache(cache_path)
    todo = [] if args.no_scrape else [(str(i), u) for i, u in zip(ids, urls) if str(i) not in cache]
    if todo:
        print(f"Scraping {len(todo):,} articles ({len(cache):,} already cached) ...")
        lock = threading.Lock()
        done = 0
        with cache_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch_article, u, args.delay, args.timeout): k for k, u in todo}
            try:
                for fut in as_completed(futs):
                    k = futs[fut]
                    res = fut.result()
                    rec = {"key": k, **res}
                    with lock:
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        fh.flush()
                        cache[k] = rec
                        done += 1
                        if done % 100 == 0 or done == len(todo):
                            ok = sum(1 for r in cache.values() if r["status"] == "ok")
                            print(f"  {done:,}/{len(todo):,} fetched this run | {ok:,} successful overall")
            except KeyboardInterrupt:
                print("Interrupted - progress is saved; re-run to resume.")
                ex.shutdown(wait=False, cancel_futures=True)

    # ---- assemble
    rows = []
    for pos, (idx, row) in enumerate(df.iterrows()):
        key = str(ids.loc[idx])
        rec = cache.get(key, {})
        scraped = (rec.get("text") or "").strip()
        use_scraped = rec.get("status") == "ok" and len(scraped) >= args.min_chars
        headline = str(row[c["headline"]]).strip() if pd.notna(row[c["headline"]]) else ""
        text = scraped if use_scraped else headline
        flag = norm_label(row[c["label"]]) if c["label"] else None

        dt = pd.to_datetime(row[c["date"]], errors="coerce", dayfirst=args.dayfirst) if c["date"] else pd.NaT
        if pd.isna(dt) and rec.get("pub"):
            dt = pd.to_datetime(rec["pub"], errors="coerce")

        rows.append({
            "index": key,
            "headline": headline,
            "article_chars": len(text),
            "article_words": len(text.split()),
            "real_fake_flag": flag,                       # 1 = real, 0 = fake
            "label_name": {1: "real", 0: "fake"}.get(flag),
            "datetime_published": dt.isoformat() if pd.notna(dt) else "",
            "article_text": text,
            "text_source": "article" if use_scraped else "headline_only",
            "url": urls.loc[idx] if c["url"] else "",
            "scrape_status": rec.get("status", "not_attempted"),
            "category": row[c["category"]] if c["category"] else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel reads Hindi/quotes right

    print(f"\nWrote {len(out):,} rows -> {args.output}")
    print("Label balance:\n", out["label_name"].value_counts(dropna=False).to_string())
    print("Text source:\n", out["text_source"].value_counts().to_string())
    print("Scrape status:\n", out["scrape_status"].value_counts().to_string())
    if (out["text_source"] == "headline_only").any():
        print("\nNOTE: rows with text_source == 'headline_only' have no scraped body. "
              "Filter on text_source == 'article' for length/wording analyses, and check "
              "whether failures differ between fake and real (dead fact-check links are "
              "more common for fake items) before drawing conclusions.")


if __name__ == "__main__":
    main()
