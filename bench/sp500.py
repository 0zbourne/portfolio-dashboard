from pathlib import Path
from io import StringIO
import pandas as pd
import requests

DATA_DIR = Path("data")
SP500_CSV = DATA_DIR / "sp500_daily.csv"

def _fetch_spy_stooq():
    """
    Free daily SPY prices (USD) from Stooq (Close; price-only proxy).
    """
    url = "https://stooq.com/q/d/l/?s=spy.us&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.rename(columns=str.lower, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).rename(columns={"close": "close_usd"})
    return df[["date", "close_usd"]]

def _fetch_fx_usd_gbp(start, end):
    """
    Daily USD->GBP from ECB via frankfurter.app for [start, end].
    """
    url = f"https://api.frankfurter.app/{start}..{end}"
    r = requests.get(url, params={"from": "USD", "to": "GBP"}, timeout=20)
    r.raise_for_status()
    data = r.json()["rates"]  # {YYYY-MM-DD: {'GBP': rate}}
    fx = (pd.DataFrame.from_dict(data, orient="index")
            .rename_axis("date").reset_index())
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    fx = fx.rename(columns={"GBP": "usd_gbp"})
    return fx.dropna(subset=["date", "usd_gbp"])

def get_sp500_daily(start: str, end: str, cache_path: Path = SP500_CSV):
    """
    Returns DataFrame with columns: date, close_usd, usd_gbp, close_gbp, daily_ret
    Also caches to data/sp500_daily.csv
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    spy = _fetch_spy_stooq()
    spy = spy[(spy["date"] >= pd.to_datetime(start)) & (spy["date"] <= pd.to_datetime(end))]
    fx = _fetch_fx_usd_gbp(start, end)
    df = pd.merge(spy, fx, on="date", how="inner").sort_values("date")
    df["close_gbp"] = df["close_usd"] * df["usd_gbp"]
    df["daily_ret"] = df["close_gbp"].pct_change()
    df.to_csv(cache_path, index=False)
    return df
