# stdlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from jobs.fundamentals import ensure_fundamentals, load_fundamentals

# ---- FORCE NUMPY BACKEND (disable Arrow/StrDtype) ----
try:
    pd.options.mode.dtype_backend = "numpy"
except Exception:
    pass
try:
    pd.options.mode.string_storage = "python"
except Exception:
    pass

# ---- CURRENCY DETECTION (integrated) ----
CURRENCY_CACHE_PATH = Path("data") / "currency_cache.json"

def _load_currency_cache() -> dict:
    """Load cached currency data from file."""
    if CURRENCY_CACHE_PATH.exists():
        try:
            return json.loads(CURRENCY_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_currency_cache(data: dict):
    """Save currency cache to file."""
    CURRENCY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENCY_CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _get_yf_symbol_from_t212(t212_ticker: str) -> str | None:
    """Convert T212 ticker to yfinance symbol for currency lookup."""
    t = (t212_ticker or "").strip().upper()
    
    # US stocks - no suffix
    if "_US_" in t:
        return t.split("_")[0]
    
    # UK stocks - add .L suffix
    core = t.replace("_GBX", "").replace("_GB", "").replace("_EQ", "")
    core = core.split("_")[0]
    
    # Strip trailing 'L' if present (e.g., "RMVL" -> "RMV")
    if core.endswith("L") and len(core) >= 4:
        core = core[:-1]
    
    if core and core.isalpha() and 1 <= len(core) <= 5:
        return f"{core}.L"
    
    return None

@st.cache_data(ttl=24 * 3600)
def get_yf_currency_cached(ysym: str) -> str:
    """
    Query yfinance for the actual currency of a ticker.
    Returns: "GBX" (pence), "GBP" (pounds), "USD", "EUR", etc.
    """
    if not ysym:
        return "USD"
    
    # Check file cache first
    cache = _load_currency_cache()
    if ysym in cache:
        return cache[ysym]
    
    # Try to import yfinance
    try:
        import yfinance as yf
    except Exception:
        # Fallback to suffix-based guess
        return "GBX" if ysym.endswith(".L") else "USD"
    
    try:
        ticker = yf.Ticker(ysym)
        info = ticker.info or {}
        raw_ccy = info.get("currency", "")
        
        # Normalize yfinance currency codes
        ccy_upper = str(raw_ccy).upper().strip()
        
        # yfinance returns 'GBp' for pence, 'GBP' for pounds
        if ccy_upper in ("GBX", "GBP", "GBPOUND", "PENCE", "PENNY", "GB PENCE"):
            result = "GBX"
        elif ccy_upper == "GBP":
            result = "GBP"
        elif ccy_upper in ("USD", "US DOLLAR"):
            result = "USD"
        elif ccy_upper in ("EUR", "EURO"):
            result = "EUR"
        elif ccy_upper:
            result = ccy_upper
        else:
            # Fallback to suffix-based guess
            result = "GBX" if ysym.endswith(".L") else "USD"
        
        # Cache the result
        cache[ysym] = result
        _save_currency_cache(cache)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
        
        return result
        
    except Exception:
        # Fallback to suffix-based guess
        result = "GBX" if ysym.endswith(".L") else "USD"
        cache[ysym] = result
        _save_currency_cache(cache)
        return result

def convert_price_to_gbp(price: float, currency: str, fx_rate: float = 1.0) -> float:
    """
    Convert a price to GBP based on its currency.
    
    Args:
        price: The raw price from yfinance/T212
        currency: "GBX" (pence), "GBP", "USD", "EUR", etc.
        fx_rate: USD to GBP rate (for USD conversions)
    
    Returns:
        Price in GBP
    """
    if currency == "GBX":
        return price / 100.0  # Pence to pounds
    elif currency == "GBP":
        return price  # Already in pounds
    elif currency == "USD":
        return price * fx_rate
    elif currency == "EUR":
        return price * fx_rate * 0.85  # Rough EUR/GBP approximation
    else:
        return price  # Unknown, assume GBP

# ---- END CURRENCY DETECTION ----

def _freshness(path: Path) -> str:
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "unknown"

def fetch_to_file(url: str, out_path: Path, timeout: int = 20):
    """
    Fetch from Trading 212 API and cache to file.
    On failure, return existing cache if present.
    """
    headers = _auth_header()
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        st.sidebar.success(f"Fetched {out_path.name}")
        return data
    except Exception as e:
        st.sidebar.error(f"Fetch failed for {url}: {e}")
        if out_path.exists():
            try:
                return json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None

def _auth_header() -> dict:
    """
    Trading212 auth: Basic base64(KEY:SECRET) or Apikey fallback.
    """
    key = (os.getenv("T212_API_KEY") or "").strip() or str(st.secrets.get("T212_API_KEY", "")).strip()
    secret = (os.getenv("T212_API_SECRET") or "").strip() or str(st.secrets.get("T212_API_SECRET", "")).strip()

    if key and secret:
        import base64
        token = base64.b64encode(f"{key}:{secret}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}", "Accept": "application/json"}

    if key:
        if key.lower().startswith("apikey "):
            return {"Authorization": key, "Accept": "application/json"}
        return {"Authorization": f"Apikey {key}", "Accept": "application/json"}

    return {}

BASE = os.getenv("T212_API_BASE", "https://live.trading212.com")

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

DATA = Path("data") / "portfolio.json"
OPEN_FILE = Path("data") / "open_prices.json"
ACC_FILE = Path("data") / "account.json"
NAV_CSV = Path("data") / "nav_daily.csv"
REPORT = Path("data") / "backfill_report.json"
OVERRIDES_PATH = Path("data") / "ticker_overrides.json"

def _load_open_prices():
    if OPEN_FILE.exists():
        try:
            return json.loads(OPEN_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_open_prices(dct):
    OPEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPEN_FILE.write_text(json.dumps(dct, indent=2), encoding="utf-8")

def _extract_cash_from_json(obj):
    preferred = {
        "free", "freecash", "free_funds",
        "availablecash", "available_funds",
        "cashbalance", "cash_balance", "available"
    }
    deny = {"id", "total", "invested", "ppl", "result", "blocked"}
    found = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = str(k).lower()
                if "piecash" in lk or lk in deny:
                    continue
                newp = f"{path}.{k}" if path else k
                if isinstance(v, (dict, list)):
                    walk(v, newp)
                else:
                    last = newp.split(".")[-1].lower()
                    if last in preferred:
                        try:
                            fv = float(v)
                            if fv >= 0:
                                found.append((newp, fv))
                        except Exception:
                            pass
        elif isinstance(o, list):
            for i, it in enumerate(o):
                walk(it, f"{path}[{i}]")

    walk(obj)
    if found:
        p, val = found[0]
        return val, p
    return None, None

def _fetch_json_quiet(url: str, timeout: int = 10):
    try:
        hdrs = _auth_header()
        r = requests.get(url, headers=hdrs, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def get_cash_gbp(base_url: str) -> tuple[float | None, str]:
    try:
        if ACC_FILE.exists() and (datetime.now(timezone.utc).timestamp() - ACC_FILE.stat().st_mtime) < 6*3600:
            data = json.loads(ACC_FILE.read_text(encoding="utf-8"))
            val, path = _extract_cash_from_json(data)
            if val is not None:
                return val, f"{ACC_FILE.name}:{path}"
    except Exception:
        pass

    endpoints = [
        "/api/v0/equity/account/info",
        "/api/v0/account/info",
        "/api/v0/accounts",
        "/api/v0/equity/account/cash",
        "/api/v0/account/cash",
    ]
    for ep in endpoints:
        url = f"{base_url}{ep}"
        data = _fetch_json_quiet(url)
        if isinstance(data, (dict, list)):
            val, path = _extract_cash_from_json(data)
            if val is not None:
                try:
                    ACC_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return val, f"{ep}:{path}"
    return None, "unavailable"

def _anchor_date_iso():
    d = datetime.now(timezone.utc).date()
    if d.weekday() == 5:
        d = d - timedelta(days=1)
    elif d.weekday() == 6:
        d = d - timedelta(days=2)
    return d.isoformat()

def ensure_today_open_prices(df):
    today = _anchor_date_iso()
    store = _load_open_prices()
    day_bucket = store.get(today, {})

    for _, row in df.iterrows():
        sym = str(row["symbol"])
        p = row.get("price_native")
        if pd.notna(p) and sym not in day_bucket:
            day_bucket[sym] = float(p)

    store[today] = day_bucket
    _save_open_prices(store)
    return store, today

@st.cache_data
def load_portfolio(cache_bust: tuple):
    with open(DATA, "r", encoding="utf-8") as f:
        items = json.load(f)

    df = pd.DataFrame(items)
    df.rename(
        columns={
            "ticker": "symbol",
            "quantity": "shares",
            "currentPrice": "price_raw",
        },
        inplace=True,
    )

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    df["price_raw"] = pd.to_numeric(df["price_raw"], errors="coerce")

    # DO NOT use threshold guessing - currency fetched from yfinance
    df["price_gbp_guess"] = df["price_raw"]  # Placeholder, will be replaced
    df["market_value_gbp_guess"] = df["shares"] * df["price_gbp_guess"]
    return df

HIST_FROM = "1970-01-01"
HIST_TO = "2100-01-01"

# --- Refresh JSONs on each run ---
fetch_to_file(f"{BASE}/api/v0/equity/portfolio", Path("data/portfolio.json"))
fetch_to_file(
    f"{BASE}/api/v0/history/transactions?from={HIST_FROM}&to={HIST_TO}",
    Path("data/transactions.json")
)
fetch_to_file(
    f"{BASE}/api/v0/history/dividends?from={HIST_FROM}&to={HIST_TO}",
    Path("data/dividends.json")
)

for p in [Path("data/portfolio.json"), Path("data/transactions.json"), Path("data/dividends.json")]:
    try:
        if p.stat().st_size < 10:
            st.sidebar.warning(f"{p.name} looks empty. Check API key/permissions and base URL.")
    except FileNotFoundError:
        st.sidebar.warning(f"{p.name} not found after fetch.")

if not DATA.exists() or DATA.stat().st_size < 10:
    st.error(
        "portfolio.json missing/empty after fetch. "
        "On Streamlit Cloud you must set a valid T212_API_KEY with permissions."
    )
    st.stop()

_stat = DATA.stat()
df = load_portfolio((_stat.st_mtime, _stat.st_size))

with st.sidebar.expander("Debug: portfolio columns", expanded=False):
    st.write(sorted(df.columns.tolist()))
    st.write(df.head(3))

# ---- Company / Instrument name ----
name_like_cols = [
    "name", "instrument.name", "displayName", "display_name",
    "company", "title", "instrumentName", "instrument_name", "security.name"
]
name_col = next((c for c in df.columns if c in name_like_cols), None)

if name_col:
    df["company"] = df[name_col].astype(str)
else:
    def _load_instrument_names(path=Path("data") / "instruments.json"):
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("items", raw)
        m = pd.json_normalize(items)
        tcol = next((c for c in m.columns if "ticker" in c.lower()), None)
        ncol = next((c for c in m.columns if "name" in c.lower()), None)
        if tcol and ncol:
            m = m[[tcol, ncol]].rename(columns={tcol: "symbol", ncol: "company"})
            return m
        return None

    name_map = _load_instrument_names()

    builtin_map = {
        "GOOGL_US_EQ": "Alphabet Inc. (Class A)",
        "MA_US_EQ": "Mastercard Inc.",
        "SPXLI_EQ": "S&P 500 UCITS ETF (Acc)",
        "GAWI_EQ": "Gawain Plc",
    }
    df["company"] = df["symbol"].map(builtin_map)

    if name_map is not None:
        df = df.merge(name_map, on="symbol", how="left", suffixes=("", "_from_map"))
        df["company"] = df["company"].combine_first(df["company_from_map"])
        if "company_from_map" in df.columns:
            df.drop(columns=["company_from_map"], inplace=True)

    def _pretty(sym: str) -> str:
        s = str(sym)
        for tag in ["_US_EQ", "_EQ", "_US", "_GBX", "_GB"]:
            s = s.replace(tag, "")
        return s.replace("_", " ").strip()
    df["company"] = df["company"].fillna(df["symbol"].apply(_pretty))

@st.cache_data
def load_true_avg_cost(path: Path = Path("data") / "transactions.json"):
    if not path.exists():
        return None, "File not found"

    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items", raw)
    if not isinstance(items, list) or len(items) == 0:
        return None, "No items in transactions.json"

    t = pd.json_normalize(items)

    action_col = next((c for c in t.columns if c.lower() in {"action", "type"}), None)
    ticker_col = next((c for c in t.columns if "ticker" in c.lower()), None)
    qty_col = next((c for c in t.columns if "quantity" in c.lower() or "shares" in c.lower()), None)
    total_col = next((c for c in t.columns if "totalamount" in c.replace("_", "").lower() or c.lower() == "total"), None)
    charge_col = next((c for c in t.columns if "charge" in c.lower()), None)
    stamp_col = next((c for c in t.columns if "stamp" in c.lower()), None)

    if any(c is None for c in [action_col, ticker_col, qty_col, total_col]):
        return None, f"Unexpected schema. Columns: {list(t.columns)}"

    for c in [qty_col, total_col, charge_col, stamp_col]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    if charge_col not in t.columns:
        t[charge_col] = 0.0
    if stamp_col not in t.columns:
        t[stamp_col] = 0.0

    buys = t[t[action_col].astype(str).str.contains("buy", case=False, na=False)].copy()
    if buys.empty:
        return None, "No buy transactions found."

    buys["spend_gbp"] = buys[total_col].fillna(0) + buys[charge_col].fillna(0) + buys[stamp_col].fillna(0)

    agg = (
        buys.groupby(buys[ticker_col])
        .agg(shares_bought=(qty_col, "sum"), total_spend_gbp=("spend_gbp", "sum"))
        .reset_index()
    )
    agg.rename(columns={ticker_col: "symbol"}, inplace=True)
    agg["true_avg_cost_gbp"] = agg["total_spend_gbp"] / agg["shares_bought"]
    return agg, None

@st.cache_data(ttl=6 * 3600)
def get_usd_gbp_rate():
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "GBP"},
            timeout=8,
        )
        r.raise_for_status()
        rate = float(r.json()["rates"]["GBP"])
        return rate, "frankfurter.app (ECB)", datetime.now(timezone.utc).isoformat()
    except Exception:
        pass
    return None, None, None

# ---- Sidebar: FX ----
st.sidebar.header("Settings")

auto_rate, rate_src, fetched_at = get_usd_gbp_rate()
if auto_rate is None:
    st.sidebar.error("FX fetch failed. Using manual override.")
    usd_to_gbp = st.sidebar.number_input("USD → GBP (manual)", value=0.78, min_value=0.5, max_value=1.5, step=0.01)
else:
    st.sidebar.metric("USD → GBP (auto)", f"{auto_rate:.4f}", help=f"Source: {rate_src}\nFetched: {fetched_at}")
    if st.sidebar.toggle("Override FX rate", value=False, key="fx_override"):
        usd_to_gbp = st.sidebar.number_input(
            "USD → GBP (override)", value=float(f"{auto_rate:.4f}"), min_value=0.5, max_value=1.5, step=0.01
        )
    else:
        usd_to_gbp = auto_rate

with st.sidebar.expander("Day change settings", expanded=False):
    if st.button("Reset today's open prices"):
        _store = _load_open_prices()
        _store[_anchor_date_iso()] = {}
        _save_open_prices(_store)
        st.success("Today's open prices reset. Reload to re-anchor.")

# ---- FETCH CURRENCIES FOR ALL TICKERS ----
# This replaces the broken threshold-based guessing
ticker_currencies = {}
with st.spinner("Fetching currency info from yfinance..."):
    for sym in df["symbol"].unique():
        ysym = _get_yf_symbol_from_t212(sym)
        if ysym:
            ticker_currencies[sym] = get_yf_currency_cached(ysym)
        else:
            ticker_currencies[sym] = "GBP"  # Default fallback

# Show currency mapping in debug
with st.sidebar.expander("Currency mapping", expanded=False):
    for sym, ccy in sorted(ticker_currencies.items()):
        ysym = _get_yf_symbol_from_t212(sym) or "N/A"
        st.caption(f"{sym} → {ysym} → {ccy}")

def gbp_price(row):
    """Convert price to GBP using ACTUAL currency from yfinance."""
    p = row["price_raw"]
    sym = str(row["symbol"])
    if pd.isna(p):
        return None
    
    ccy = ticker_currencies.get(sym, "GBP")
    return convert_price_to_gbp(float(p), ccy, usd_to_gbp)

df["price_gbp"] = df.apply(gbp_price, axis=1)
df["market_value_gbp"] = df["shares"] * df["price_gbp"]

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Positions", f"{len(df):,}")
c2.metric("Total Shares", f"{int(df['shares'].sum()):,}")
c3.metric("Market Value (GBP)", f"£{df['market_value_gbp'].sum():,.0f}")

st.caption("Currency fetched from yfinance (GBX = pence, divided by 100; USD converted via FX rate).")

# =======================
# Dividends loader
# =======================
DIV_FILE = Path("data") / "dividends.json"

@st.cache_data
def load_dividends_t212(path=DIV_FILE):
    if not path.exists():
        return None, "File not found"

    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items", [])
    if not isinstance(items, list) or len(items) == 0:
        return None, "No items in dividends.json"

    df_div = pd.json_normalize(items)

    if "type" in df_div.columns:
        allowed = {"DIVIDEND", "DIVIDEND_CASH", "CASH_DIVIDEND"}
        df_div = df_div[df_div["type"].astype(str).str.upper().isin(allowed)]

    if "reference" in df_div.columns:
        df_div = df_div[~df_div["reference"].astype(str).str.contains("interest", case=False, na=False)]

    if "paidOn" not in df_div.columns or "amount" not in df_div.columns:
        return None, f"Unexpected columns: {list(df_div.columns)}"

    paid = df_div["paidOn"].astype(str).str.strip().str.replace("Z", "", regex=False)
    dt = pd.to_datetime(paid, errors="coerce", utc=True)
    if dt.isna().all():
        dt = pd.to_datetime(paid, format="%Y-%m-%d", errors="coerce", utc=True)
    if dt.isna().all():
        dt = pd.to_datetime(paid, unit="ms", errors="coerce", utc=True)

    df_div["dt"] = dt
    if df_div["dt"].isna().all():
        return None, "Could not parse 'paidOn' dates."

    out = (
        pd.DataFrame({
            "dt": df_div["dt"],
            "ticker": df_div.get("ticker"),
            "amount_gbp": pd.to_numeric(df_div["amount"], errors="coerce"),
        })
        .dropna(subset=["dt", "amount_gbp"])
        .sort_values("dt")
    )
    return out, None

# ---------- Holdings ----------
st.subheader("Holdings")

avg_candidates = [c for c in df.columns if "average" in c.lower() and "price" in c.lower()]
df["avg_cost_raw"] = pd.to_numeric(df[avg_candidates[0]], errors="coerce") if avg_candidates else None

def _ccy(sym: str) -> str:
    """Return display currency (USD or GBP)."""
    return ticker_currencies.get(sym, "GBP")

df["ccy"] = df["symbol"].apply(_ccy)
df["avg_cost_raw"] = pd.to_numeric(df["avg_cost_raw"], errors="coerce")

def _price_native(row):
    """Get native price (no conversion)."""
    p = row["price_raw"]
    if pd.isna(p):
        return np.nan
    ccy = ticker_currencies.get(row["symbol"], "GBP")
    # If GBX, show as pence value for display (divide later for GBP)
    if ccy == "GBX":
        return float(p) / 100.0  # Show in pounds for consistency
    elif ccy == "USD":
        return float(p)
    else:
        return float(p)

def _avg_native(row):
    """Get native average cost (no conversion)."""
    x = row["avg_cost_raw"]
    if pd.isna(x):
        return np.nan
    ccy = ticker_currencies.get(row["symbol"], "GBP")
    if ccy == "GBX":
        return float(x) / 100.0  # Convert pence to pounds
    elif ccy == "USD":
        return float(x)
    else:
        return float(x)

df["price_native"] = df.apply(_price_native, axis=1)
df["avg_cost_native"] = df.apply(_avg_native, axis=1)

def _find_candidates(cols, must_have_any, also_any=None):
    out = []
    low = {c: c.lower() for c in cols}
    for c, lc in low.items():
        if any(tok in lc for tok in must_have_any) and (not also_any or any(tok in lc for tok in also_any)):
            out.append(c)
    return out

day_abs_candidates = (
    _find_candidates(df.columns, ["day", "today"], ["pnl", "pl", "change", "chg", "diff", "return"])
    + _find_candidates(df.columns, ["change"], ["day", "today"])
)

day_pct_candidates = (
    _find_candidates(df.columns, ["day", "today"], ["pct", "percent", "%", "return"])
    + _find_candidates(df.columns, ["changepct", "percent", "%"])
)

with st.sidebar.expander("Map day-change fields", expanded=False):
    sel_abs = st.selectbox(
        "Day Change £ column (per-share OR position)",
        options=["<auto>"] + day_abs_candidates, index=0
    )
    sel_pct = st.selectbox(
        "Day Change % column",
        options=["<auto>"] + day_pct_candidates, index=0
    )

div_tbl, _div_err = load_dividends_t212(Path("data") / "dividends.json")
if _div_err is None and div_tbl is not None:
    by_ticker = (
        div_tbl.groupby("ticker", as_index=False)["amount_gbp"].sum()
        .rename(columns={"ticker": "symbol", "amount_gbp": "dividends_gbp"})
    )
    df = df.merge(by_ticker, on="symbol", how="left")
else:
    df["dividends_gbp"] = None

true_costs, tc_err = load_true_avg_cost(Path("data") / "transactions.json")
if tc_err is None and true_costs is not None:
    df = df.merge(true_costs[["symbol", "true_avg_cost_gbp"]], on="symbol", how="left")
    df["cost_basis_gbp"] = df["true_avg_cost_gbp"] * df["shares"]
else:
    df["true_avg_cost_gbp"] = None
    df["cost_basis_gbp"] = None

df["total_value_gbp"] = pd.to_numeric(df["market_value_gbp"], errors="coerce")
portfolio_total = float(df["total_value_gbp"].sum())
df["weight_pct"] = np.where(
    portfolio_total > 0,
    (df["total_value_gbp"] / portfolio_total) * 100.0,
    0.0,
)

# -------------------------
# Allocation (incl. Cash)
# -------------------------
st.subheader("Allocation (including Cash)")
st.caption(f"As of { _freshness(Path('data') / 'account.json') }")

cash_gbp, cash_src = get_cash_gbp(BASE)
has_cash = (cash_gbp is not None) and (cash_gbp >= 0)

total_with_cash = portfolio_total + (cash_gbp if has_cash else 0.0)

alloc = df[["symbol", "company", "total_value_gbp"]].copy()
alloc = alloc.dropna(subset=["total_value_gbp"]).sort_values("total_value_gbp", ascending=False)

rows = []
for r in alloc.itertuples():
    w = (float(r.total_value_gbp) / total_with_cash) if total_with_cash > 0 else 0.0
    label = str(r.symbol) if pd.notna(r.symbol) and str(r.symbol).strip() else str(r.company)
    rows.append({"label": label, "weight": w})

if has_cash:
    rows.append({"label": "Cash", "weight": (cash_gbp / total_with_cash) if total_with_cash > 0 else 0.0})

pie = pd.DataFrame(rows)

if not pie.empty:
    pie["pct"] = (pie["weight"] * 100.0)
    pie["pct_r"] = pie["pct"].round(1)
    diff = 100.0 - pie["pct_r"].sum()
    if abs(diff) >= 0.1:
        idx = pie["pct"].idxmax()
        pie.loc[idx, "pct_r"] = pie.loc[idx, "pct_r"] + diff
    pie["pct_r"] = pie["pct_r"].clip(lower=0)
else:
    pie["pct_r"] = []

if pie["pct_r"].sum() <= 0:
    st.info("No holdings to plot.")
else:
    chart = (
        alt.Chart(pie)
        .mark_arc(innerRadius=90)
        .encode(
            theta=alt.Theta("pct_r:Q", stack=True, title=None),
            order=alt.Order("pct_r:Q", sort="descending"),
            color=alt.Color("label:N", legend=alt.Legend(title=None, orient="right")),
            tooltip=[
                alt.Tooltip("label:N", title="Position"),
                alt.Tooltip("pct_r:Q", title="Weight", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")

caption = "Weights by current market value"
if has_cash:
    cash_pct = (cash_gbp / total_with_cash * 100.0) if total_with_cash > 0 else 0.0
    caption += f" • Cash: £{cash_gbp:,.0f} ({cash_pct:.1f}%) • Source: {cash_src}"
else:
    caption += " • Cash unavailable (positions only)"
st.caption(caption + ".")

# --- Day change calculation ---
_prev_close_candidates = [
    "previousClose", "prevClose", "lastClose", "closePrevious",
    "pricePrevClose", "previous_close", "priorClose"
]
prev_col = next((c for c in _prev_close_candidates if c in df.columns), None)

df["day_change_gbp"] = float("nan")
df["day_change_pct"] = float("nan")

if prev_col:
    def _prev_native(row):
        v = row[prev_col]
        if v is None or pd.isna(v):
            return None
        v = float(v)
        ccy = ticker_currencies.get(row["symbol"], "GBP")
        if ccy == "GBX":
            return v / 100.0
        return v

    df["prev_close_native"] = df.apply(_prev_native, axis=1)
    df["day_change_native"] = df["price_native"] - df["prev_close_native"]
else:
    store, today_key = ensure_today_open_prices(df)
    today_opens = store.get(today_key, {})

    def _open_native(row):
        p = today_opens.get(str(row["symbol"]))
        return float(p) if p is not None else float("nan")

    df["prev_close_native"] = df.apply(_open_native, axis=1)
    df["day_change_native"] = df["price_native"] - df["prev_close_native"]

def _native_to_gbp(row, v):
    if pd.isna(v):
        return float("nan")
    ccy = ticker_currencies.get(row["symbol"], "GBP")
    if ccy == "USD":
        return float(v) * usd_to_gbp
    return float(v)

df["day_change_gbp"] = df.apply(
    lambda r: r["shares"] * _native_to_gbp(r, r["day_change_native"]),
    axis=1
)

df["day_change_pct"] = (df["price_native"] / df["prev_close_native"] - 1.0) * 100

if 'sel_abs' in locals() and sel_abs != "<auto>" and sel_abs in df.columns:
    tmp_abs = pd.to_numeric(df[sel_abs], errors="coerce")
    try:
        med_price = pd.to_numeric(df["price_native"], errors="coerce").median()
        med_abs = tmp_abs.abs().median()
        per_share = bool(med_price and med_abs and med_abs < (3 * med_price))
    except Exception:
        per_share = False

    def _to_gbp(row, val):
        if pd.isna(val):
            return float("nan")
        scaled = (val * row["shares"]) if per_share else val
        ccy = ticker_currencies.get(row["symbol"], "GBP")
        if ccy == "USD":
            return float(scaled) * usd_to_gbp
        return float(scaled)

    df["day_change_gbp"] = df.apply(lambda r: _to_gbp(r, tmp_abs.loc[r.name]), axis=1)

if 'sel_pct' in locals() and sel_pct != "<auto>" and sel_pct in df.columns:
    df["day_change_pct"] = pd.to_numeric(df[sel_pct], errors="coerce")

df["day_change_gbp"] = pd.to_numeric(df["day_change_gbp"], errors="coerce")
df["day_change_pct"] = pd.to_numeric(df["day_change_pct"], errors="coerce")

def _gbp_from_native(row, x):
    if pd.isna(x):
        return float("nan")
    ccy = ticker_currencies.get(row["symbol"], "GBP")
    if ccy == "USD":
        return float(x) * usd_to_gbp
    return float(x)

df["true_avg_cost_gbp"] = pd.to_numeric(df["true_avg_cost_gbp"], errors="coerce")
df["dividends_gbp"] = pd.to_numeric(df["dividends_gbp"], errors="coerce")
df["cost_basis_gbp"] = df["shares"] * df["true_avg_cost_gbp"]

approx_cost_each_gbp = df.apply(lambda r: _gbp_from_native(r, r["avg_cost_native"]), axis=1)
df["cost_basis_gbp_approx"] = df["shares"] * approx_cost_each_gbp
df["cost_basis_gbp_effective"] = df["cost_basis_gbp"].combine_first(df["cost_basis_gbp_approx"])

df["capital_gains_gbp"] = pd.to_numeric(df["total_value_gbp"], errors="coerce") - pd.to_numeric(
    df["cost_basis_gbp_effective"], errors="coerce"
)

df["total_return_gbp"] = df["capital_gains_gbp"] + df["dividends_gbp"].fillna(0)

den = df["cost_basis_gbp_effective"]
df["total_return_pct"] = np.where(
    (den.notna()) & (den != 0),
    (df["total_return_gbp"] / den) * 100.0,
    np.nan,
)

df["pl_pct_native"] = (
    (df["price_native"] - df["avg_cost_native"]) / df["avg_cost_native"] * 100
)

num_cols = [
    "price_native", "avg_cost_native", "day_change_gbp", "day_change_pct",
    "total_return_gbp", "total_return_pct", "dividends_gbp",
    "capital_gains_gbp", "total_value_gbp", "weight_pct",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

view = df[[
    "company", "shares", "ccy", "price_native", "avg_cost_native",
    "day_change_gbp", "day_change_pct", "total_return_gbp", "total_return_pct",
    "dividends_gbp", "capital_gains_gbp", "total_value_gbp", "weight_pct",
]].copy()

st.dataframe(
    view.sort_values("total_value_gbp", ascending=False),
    width="stretch",
    hide_index=True,
    column_config={
        "company": "Name",
        "shares": st.column_config.NumberColumn("No. of Shares", format="%.0f"),
        "ccy": st.column_config.TextColumn("Ccy"),
        "price_native": st.column_config.NumberColumn("Live Share Price", format="%.2f"),
        "avg_cost_native": st.column_config.NumberColumn("Avg. Share Price", format="%.2f"),
        "day_change_gbp": st.column_config.NumberColumn("Day Change £", format="£%.0f"),
        "day_change_pct": st.column_config.NumberColumn("Day Change %", format="%.2f%%"),
        "total_return_gbp": st.column_config.NumberColumn("Total Return £", format="£%.0f"),
        "total_return_pct": st.column_config.NumberColumn("Total Return %", format="%.2f%%"),
        "dividends_gbp": st.column_config.NumberColumn("Dividends", format="£%.0f"),
        "capital_gains_gbp": st.column_config.NumberColumn("Capital Gains", format="£%.0f"),
        "total_value_gbp": st.column_config.NumberColumn("Holding Value", format="£%.0f"),
        "weight_pct": st.column_config.NumberColumn("Weight %", format="%.2f%%"),
    },
)

notes = []
if df["true_avg_cost_gbp"].isna().any():
    notes.append("Using **approximate cost basis** from average price for some rows (FX-adjusted). Upload transactions.json for true GBP cost.")
if df["day_change_gbp"].isna().all():
    notes.append("Day change anchored to today's first seen price (or map fields in the sidebar if your API provides them).")
if notes:
    st.caption(" • ".join(notes))

# =======================
# Portfolio Quality Strip
# =======================
try:
    wdf = df[["symbol", "total_value_gbp"]].copy()
    wdf["total_value_gbp"] = pd.to_numeric(wdf["total_value_gbp"], errors="coerce")
    wdf = wdf.dropna(subset=["total_value_gbp"])
    tot_mv = float(wdf["total_value_gbp"].sum())
    weights = {str(r.symbol): float(r.total_value_gbp) / tot_mv for r in wdf.itertuples()} if tot_mv > 0 else {}

    ensure_fundamentals(weights)
    fund = load_fundamentals()
    agg = (fund or {}).get("portfolio_weighted", {})

    st.subheader("Portfolio quality (TTM, weight-adjusted)")
    st.caption(f"As of { _freshness(Path('data') / 'fundamentals.json') }")
    k1, k2, k3, k4, k5 = st.columns(5)

    def _fmt_pct(x):
        return "N/A" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x*100:.0f}%"
    def _fmt_ic(x):
        return "N/A" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x:.1f}×"

    k1.metric("ROCE", _fmt_pct(agg.get("roce")))
    k2.metric("Gross margin", _fmt_pct(agg.get("gm")))
    k3.metric("Operating margin", _fmt_pct(agg.get("om")))
    k4.metric("Cash conversion (FCF/EBIT)", _fmt_pct(agg.get("cc")))
    k5.metric("Interest cover", _fmt_ic(agg.get("ic")))

    with st.expander("Quality audit (per-ticker)", expanded=False):
        from jobs.fundamentals import FUND_AUDIT, FUND_JSON
        st.caption(f"Data source: yfinance. Basis: TTM (fallback FY). Cache: weekly. File: {FUND_JSON}")
        if FUND_AUDIT.exists():
            st.dataframe(pd.read_csv(FUND_AUDIT), width="stretch")
        else:
            st.write("Audit file not available yet.")
except Exception as e:
    st.warning(f"Quality strip unavailable: {e}")

# ---- Render dividends timeline ----
from pandas.api.types import is_datetime64_any_dtype as is_datetime

div, err = load_dividends_t212(DIV_FILE)

st.subheader("Dividends timeline")
if err:
    st.write("Dividends (inspect)")
    st.warning(err)
else:
    if not is_datetime(div["dt"]):
        div["dt"] = pd.to_datetime(div["dt"], errors="coerce", utc=True)
    div = div.dropna(subset=["dt", "amount_gbp"]).copy()
    div["amount_gbp"] = pd.to_numeric(div["amount_gbp"], errors="coerce")
    div = div.dropna(subset=["amount_gbp"])

    years = sorted(div["dt"].dt.year.unique().tolist())
    selected_year = st.sidebar.selectbox("Year", options=["All"] + years, index=0)

    if selected_year != "All":
        div = div[div["dt"].dt.year == int(selected_year)]

    div["year_month"] = div["dt"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    monthly = div.groupby("year_month", as_index=False)["amount_gbp"].sum()
    monthly["year_month"] = monthly["year_month"].astype(str)

    st.bar_chart(monthly, x="year_month", y="amount_gbp", width="stretch")
    st.caption("Monthly dividend cash received (GBP).")

# -----------------------
# One-off NAV backfill UI
# -----------------------
with st.sidebar.expander("Backfill NAV (since 2025-01-01)", expanded=False):
    st.write("Rebuild daily NAV in GBP from **orders** + yfinance prices. Optional overrides: data/ticker_overrides.json")
    start_str = st.text_input("Start date", value="2025-01-01")

    if st.button("Run NAV backfill"):
        try:
            from jobs.backfill import backfill_nav_from_orders
            out_path = backfill_nav_from_orders(start=start_str)
            st.success(f"NAV backfill complete → {out_path}")
            rep_path = Path("data") / "backfill_report.json"
            if rep_path.exists():
                rep = json.loads(rep_path.read_text(encoding="utf-8"))
                missing = rep.get("missing_symbols", [])
                if missing:
                    st.warning(f"No price history for: {', '.join(missing)}. Add mappings in data/ticker_overrides.json.")
                else:
                    st.caption("All symbols fetched successfully.")
        except Exception as e:
            st.exception(e)
            st.caption("See data/_backfill_trace.txt for full details.")

# =======================
# Backend performance plumbing
# =======================
from jobs.snapshot import append_today_snapshot_if_missing
from pdperf.series import read_nav, daily_returns_twr, cumulative_return, cagr
from pdperf.cashflows import build_cash_flows
from bench.sp500 import get_sp500_daily

try:
    append_today_snapshot_if_missing(df)
except Exception as e:
    st.sidebar.warning(f"NAV snapshot not updated: {e}")

try:
    today_key = _anchor_date_iso()

    nav = read_nav()
    flows = build_cash_flows(Path("data") / "transactions.json")
    port_daily = daily_returns_twr(nav, flows)

    perf_start = pd.to_datetime(nav.index).min().strftime("%Y-%m-%d")
    sp = get_sp500_daily(perf_start, today_key)

    port_daily["date"] = pd.to_datetime(port_daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sp["date"] = pd.to_datetime(sp["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = (
        port_daily.merge(
            sp[["date", "daily_ret"]].rename(columns={"daily_ret": "r_bench"}),
            on="date",
            how="inner"
        )
        .dropna(subset=["r_port", "r_bench"])
        .sort_values("date")
    )

    since_start_port = cumulative_return(merged, perf_start, today_key)
    since_start_bench = float((1 + merged["r_bench"]).prod() - 1) if not merged.empty else float("nan")

    year_start = f"{datetime.now(timezone.utc).year}-01-01"
    ytd_slice = merged[(merged["date"] >= year_start) & (merged["date"] <= today_key)]
    ytd_port = float((1 + ytd_slice["r_port"]).prod() - 1) if not ytd_slice.empty else float("nan")
    ytd_bench = float((1 + ytd_slice["r_bench"]).prod() - 1) if not ytd_slice.empty else float("nan")

    if pd.notna(since_start_port) and pd.notna(since_start_bench):
        st.sidebar.caption(f"Since {perf_start}: Portfolio {since_start_port:.2%} vs S&P {since_start_bench:.2%}")
    if pd.notna(ytd_port) and pd.notna(ytd_bench):
        st.sidebar.caption(f"YTD (backend): Portfolio {ytd_port:.2%} vs S&P {ytd_bench:.2%}")

    anchor = "2025-01-01"

    plot = merged[merged["date"] >= anchor].copy()
    if plot.empty:
        st.info("Not enough data after the anchor date to draw the NAV vs S&P chart.")
    else:
        plot["Portfolio (TWR)"] = (100.0 * (1.0 + plot["r_port"]).cumprod()).astype("float64")
        plot["S&P 500 (GBP)"] = (100.0 * (1.0 + plot["r_bench"]).cumprod()).astype("float64")

        st.subheader("NAV vs S&P 500 (rebased to 100)")
        st.caption(f"As of { _freshness(NAV_CSV) }")

        plot_alt = plot.copy()
        plot_alt["date"] = pd.to_datetime(plot_alt["date"], errors="coerce")
        plot_alt = plot_alt.melt(
            id_vars=["date"],
            value_vars=["Portfolio (TWR)", "S&P 500 (GBP)"],
            var_name="series",
            value_name="index"
        )

        y_min = float(plot[["Portfolio (TWR)", "S&P 500 (GBP)"]].min().min()) * 0.95
        y_max = float(plot[["Portfolio (TWR)", "S&P 500 (GBP)"]].max().max()) * 1.05

        chart = (
            alt.Chart(plot_alt)
            .mark_line()
            .encode(
                x=alt.X("date:T", title=""),
                y=alt.Y(
                    "index:Q",
                    title="Index (rebased = 100)",
                    scale=alt.Scale(domain=[y_min, y_max], nice=False, zero=False),
                ),
                color=alt.Color("series:N", legend=alt.Legend(title=None)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("index:Q", title="Value", format=".2f"),
                ],
            )
            .properties(height=340)
        )

        st.altair_chart(chart, width="stretch")

except Exception as e:
    st.sidebar.info(f"Perf debug unavailable: {e}")

    # Debug section - show file contents
    with st.expander("Debug: Data Files", expanded=False):
        st.subheader("nav_daily.csv")
        if NAV_CSV.exists():
            nav_debug = pd.read_csv(NAV_CSV)
            st.write(f"Rows: {len(nav_debug)}")
            st.write("First 5 rows:")
            st.dataframe(nav_debug.head(5), width="stretch")
            st.write("Last 5 rows:")
            st.dataframe(nav_debug.tail(5), width="stretch")
        else:
            st.write("File not found")

        st.subheader("backfill_report.json")
        if REPORT.exists():
            report_debug = json.loads(REPORT.read_text(encoding="utf-8"))
            st.json(report_debug)
        else:
            st.write("File not found")

        st.subheader("portfolio.json (first 2 items)")
        if DATA.exists():
            port_debug = json.loads(DATA.read_text(encoding="utf-8"))
            if isinstance(port_debug, list):
                st.json(port_debug[:2])
            else:
                st.json(port_debug)
        else:
            st.write("File not found")

        # 
        st.subheader("Debug: Flow Alignment Check")
        flows = build_cash_flows(Path("data") / "transactions.json")
        nav = read_nav()
        
        st.write("NAV dates (first 10):", nav.index[:10].tolist())
        st.write("NAV dates (last 10):", nav.index[-10:].tolist())
        
        if flows is not None and not flows.empty:
            st.write("Flow dates (first 10):", flows["date"].head(10).tolist())
            st.write("Flow dates (last 10):", flows["date"].tail(10).tolist())
            
            # Check for flows that don't match NAV dates
            flow_dates = set(pd.to_datetime(flows["date"]).dt.date)
            nav_dates = set(nav.index.date)
            mismatch = flow_dates - nav_dates
            st.write("Flow dates NOT in NAV:", mismatch)

# =======================
# Debug: Aug 18 Position Analysis
# =======================
st.subheader("Debug: Aug 18 Position Analysis")
import datetime as dt

try:
    pos_path = Path("data") / "positions_cache.parquet"
    prices_path = Path("data") / "prices_cache.parquet"
    
    if pos_path.exists() and prices_path.exists():
        pos = pd.read_parquet(pos_path)
        prices = pd.read_parquet(prices_path)
        
        aug17_date = dt.date(2025, 8, 17)
        aug18_date = dt.date(2025, 8, 18)
        
        st.write("### Positions Aug 17:")
        if aug17_date in pos.index:
            row = pos.loc[aug17_date]
            st.write(row[row > 0])
        else:
            st.write("Date not found in positions")
        
        st.write("### Positions Aug 18:")
        if aug18_date in pos.index:
            row = pos.loc[aug18_date]
            st.write(row[row > 0])
        else:
            st.write("Date not found in positions")
        
        st.write("### Price Changes (Aug 18 vs Aug 17):")
        if aug17_date in prices.index and aug18_date in prices.index:
            p17 = prices.loc[aug17_date]
            p18 = prices.loc[aug18_date]
            change = (p18 / p17 - 1) * 100
            st.write(change.dropna().sort_values(ascending=False))
        else:
            st.write("Dates not found in prices")
            
        st.write("### NAV Calculation:")
        if aug17_date in pos.index and aug18_date in pos.index:
            nav17 = (pos.loc[aug17_date] * prices.loc[aug17_date]).sum()
            nav18 = (pos.loc[aug18_date] * prices.loc[aug18_date]).sum()
            st.write(f"NAV Aug 17: £{nav17:,.2f}")
            st.write(f"NAV Aug 18: £{nav18:,.2f}")
            st.write(f"Difference: £{nav18 - nav17:,.2f}")
            st.write(f"Expected: ~£115 (trading fees)")
    else:
        st.write("positions_cache.parquet or prices_cache.parquet not found")
        st.write("Run NAV backfill first to generate these files.")
        
except Exception as e:
    st.error(f"Debug error: {e}")

# =======================
# Debug: Export data files
# =======================
with st.sidebar.expander("🔧 Debug: Export Data", expanded=False):
    st.markdown("### Data Files")

    # nav_daily.csv
    if NAV_CSV.exists():
        nav_data = NAV_CSV.read_text(encoding="utf-8")
        st.download_button(
            label="📥 Download nav_daily.csv",
            data=nav_data,
            file_name="nav_daily.csv",
            mime="text/csv",
        )
        st.caption(f"Rows: {len(nav_data.splitlines()) - 1}")
    else:
        st.write("nav_daily.csv not found")

    # transactions.json
    tx_path = Path("data") / "transactions.json"
    if tx_path.exists():
        tx_data = tx_path.read_text(encoding="utf-8")
        st.download_button(
            label="📥 Download transactions.json",
            data=tx_data,
            file_name="transactions.json",
            mime="application/json",
        )
        st.caption(f"Size: {len(tx_data):,} bytes")
    else:
        st.write("transactions.json not found")

    # backfill_report.json
    if REPORT.exists():
        rep_data = REPORT.read_text(encoding="utf-8")
        st.download_button(
            label="📥 Download backfill_report.json",
            data=rep_data,
            file_name="backfill_report.json",
            mime="application/json",
        )
    else:
        st.write("backfill_report.json not found")

    # ticker_overrides.json
    if OVERRIDES_PATH.exists():
        ov_data = OVERRIDES_PATH.read_text(encoding="utf-8")
        st.download_button(
            label="📥 Download ticker_overrides.json",
            data=ov_data,
            file_name="ticker_overrides.json",
            mime="application/json",
        )
    else:
        st.write("ticker_overrides.json not found")

    st.markdown("---")
    st.markdown("### Inspect NAV around Feb 2025")

    if NAV_CSV.exists():
        nav_df = pd.read_csv(NAV_CSV)
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        feb_slice = nav_df[(nav_df["date"] >= "2025-02-20") & (nav_df["date"] <= "2025-02-28")]
        if not feb_slice.empty:
            st.dataframe(feb_slice, width="stretch", hide_index=True)
            st.caption(f"NAV range: £{feb_slice['nav_gbp'].min():,.0f} - £{feb_slice['nav_gbp'].max():,.0f}")
        else:
            st.write("No data for Feb 2025")

# =======================
# Debug: NAV Breakdown
# =======================
with st.sidebar.expander("🔍 Debug: NAV Breakdown", expanded=False):
    st.markdown("### Inspect NAV Calculation")
    
    if NAV_CSV.exists():
        nav_df = pd.read_csv(NAV_CSV)
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        
        # Date picker
        available_dates = nav_df["date"].dt.strftime("%Y-%m-%d").tolist()
        default_date = "2025-02-24" if "2025-02-24" in available_dates else available_dates[-1] if available_dates else None
        
        selected_date = st.selectbox(
            "Select date to inspect:",
            options=available_dates,
            index=available_dates.index(default_date) if default_date else 0,
            key="nav_breakdown_date"
        )
        
        if st.button("Load Breakdown", key="load_breakdown_btn"):
            from jobs.backfill import get_nav_breakdown
            
            breakdown = get_nav_breakdown(selected_date)
            
            if breakdown is not None:
                st.session_state["nav_breakdown_df"] = breakdown
                st.session_state["nav_breakdown_date"] = selected_date
            else:
                st.error("No breakdown data available. Run NAV backfill first.")
        
        if "nav_breakdown_df" in st.session_state:
            bd = st.session_state["nav_breakdown_df"]
            st.markdown(f"**NAV Breakdown for {st.session_state.get('nav_breakdown_date', selected_date)}**")
            
            st.dataframe(
                bd,
                width="stretch",
                hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("T212 Ticker"),
                    "yf_symbol": st.column_config.TextColumn("Yahoo Symbol"),
                    "currency": st.column_config.TextColumn("Currency"),
                    "shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                    "price_gbp": st.column_config.NumberColumn("Price (GBP)", format="£%.4f"),
                    "value_gbp": st.column_config.NumberColumn("Value (GBP)", format="£%.2f"),
                }
            )
            
            # Export button
            csv_data = bd.to_csv(index=False)
            st.download_button(
                label="📥 Download Breakdown CSV",
                data=csv_data,
                file_name=f"nav_breakdown_{selected_date}.csv",
                mime="text/csv",
            )
            
            # Show NAV from file for comparison
            nav_on_date = nav_df[nav_df["date"].dt.strftime("%Y-%m-%d") == selected_date]
            if not nav_on_date.empty:
                st.caption(f"NAV from file: £{nav_on_date['nav_gbp'].iloc[0]:,.2f}")
    else:
        st.write("Run NAV backfill first to enable breakdown.")
