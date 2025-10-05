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
    DAILY (business-day) S&P proxy in GBP with stable returns.
    Returns columns: date, close_usd, usd_gbp, close_gbp, daily_ret (all float64).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Business-day calendar
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    cal = pd.bdate_range(s, e).date  # business days only

    # 2) SPY (USD) from Stooq → reindex to daily calendar, ffill/bfill
    spy = _fetch_spy_stooq().copy()
    spy["date"] = pd.to_datetime(spy["date"]).dt.date
    spy = (spy.set_index("date")
              .reindex(cal)
              .ffill()
              .bfill())
    spy["close_usd"] = pd.to_numeric(spy["close_usd"], errors="coerce").astype("float64")

    # 3) USD→GBP FX from ECB → reindex to daily calendar, ffill/bfill
    fx = _fetch_fx_usd_gbp(start, end).copy()
    fx["date"] = pd.to_datetime(fx["date"]).dt.date
    fx = (fx.set_index("date")
            .reindex(cal)
            .ffill()
            .bfill())
    fx["usd_gbp"] = pd.to_numeric(fx["usd_gbp"], errors="coerce").astype("float64")

    # 4) Compose daily GBP series and daily returns
    df = pd.DataFrame(index=cal)
    df["close_usd"] = spy["close_usd"].astype("float64")
    df["usd_gbp"]   = fx["usd_gbp"].astype("float64")
    df["close_gbp"] = (df["close_usd"] * df["usd_gbp"]).astype("float64")
    df["daily_ret"] = df["close_gbp"].pct_change().fillna(0.0).astype("float64")

    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df.to_csv(cache_path, index=False)
    return df