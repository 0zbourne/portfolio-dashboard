# stdlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

# ---- FORCE NUMPY BACKEND (disable Arrow/StrDType) ----
try:
    pd.options.mode.dtype_backend = "numpy"     # pandas >= 2.1
except Exception:
    pass
try:
    pd.options.mode.string_storage = "python"   # avoid pyarrow-backed strings
except Exception:
    pass

# --- Fetch helper to refresh local JSONs from T212 API ---
def fetch_to_file(url: str, out_path: Path):
    # headers = {"Authorization": f"Apikey {API_KEY}", "Accept": "application/json"}
    headers = {"Authorization": API_KEY, "Accept": "application/json"} # <-- adjust if your API wants 'Apikey ' or similar
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        st.sidebar.success(f"Fetched {out_path.name}")
        return data
    except Exception as e:
        st.sidebar.error(f"Fetch failed for {url}: {e}")
        return None

API_KEY = os.getenv("T212_API_KEY")
if not API_KEY:
    st.error("Missing T212 API key — set it with setx T212_API_KEY ...")
    st.stop()

BASE = os.getenv("T212_API_BASE", "https://live.trading212.com") # use https://demo.trading212.com for practice  # practice API host

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

DATA = Path("data") / "portfolio.json"

# --- Persist & read "opening price for the day" so we can compute day change ---
OPEN_FILE = Path("data") / "open_prices.json"

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

def _anchor_date_iso():
    """
    Use UTC date, but roll back to last business day if it's Saturday/Sunday.
    Sat = 5, Sun = 6
    """
    d = datetime.utcnow().date()
    if d.weekday() == 5:      # Saturday
        d = d.replace(day=d.day)  # no-op for clarity
        d = d - timedelta(days=1)  # Friday
    elif d.weekday() == 6:    # Sunday
        d = d - timedelta(days=2)  # Friday
    return d.isoformat()

def ensure_today_open_prices(df):
    """
    Ensure we have an 'open price' for each symbol for today's date.
    If missing, use current price_native as the 'open' anchor.
    Returns (store_dict, today_key).
    """
    today = _anchor_date_iso()
    store = _load_open_prices()
    day_bucket = store.get(today, {})

    # If we don't yet have an opening price for a symbol, record it now.
    for _, row in df.iterrows():
        sym = str(row["symbol"])
        p = row.get("price_native")
        if pd.notna(p) and sym not in day_bucket:
            day_bucket[sym] = float(p)

    store[today] = day_bucket
    _save_open_prices(store)
    return store, today

# Portfolio loader
@st.cache_data
def load_portfolio(cache_bust: tuple):
    """
    Cache-busted by the portfolio.json file's (mtime, size).
    Passing this tuple forces Streamlit to reload when the file changes.
    """
    with open(DATA, "r", encoding="utf-8") as f:
        items = json.load(f)

    df = pd.DataFrame(items)

    # Canonical column names
    df.rename(
        columns={
            "ticker": "symbol",
            "quantity": "shares",
            "currentPrice": "price_raw",
        },
        inplace=True,
    )

    # ---- Strong dtype coercion (prevents '0', None, '' oddities)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    df["price_raw"] = pd.to_numeric(df["price_raw"], errors="coerce")

    # ---- Safe GBX → GBP guess (vectorised, NaN-safe)
    # Treat values > 1000 as pence (e.g., 14680 → £146.80)
    df["price_gbp_guess"] = np.where(
        (df["price_raw"].notna()) & (df["price_raw"] > 1000),
        df["price_raw"] / 100.0,
        df["price_raw"],
    )

    # Simple MV using the guessed GBP (used only as a rough placeholder)
    df["market_value_gbp_guess"] = df["shares"] * df["price_gbp_guess"]

    return df

# --- Refresh JSONs on each run (or comment out if you want manual refresh only) ---
HIST_FROM = "1970-01-01"
HIST_TO   = "2100-01-01" # Wide date range so history isn't empty

fetch_to_file(f"{BASE}/api/v0/equity/portfolio", Path("data/portfolio.json"))
fetch_to_file(
    f"{BASE}/api/v0/history/transactions?from={HIST_FROM}&to={HIST_TO}",
    Path("data/transactions.json")
)
fetch_to_file(
    f"{BASE}/api/v0/history/dividends?from={HIST_FROM}&to={HIST_TO}",
    Path("data/dividends.json")
)

# Warning if files are empty
for p in [Path("data/portfolio.json"), Path("data/transactions.json"), Path("data/dividends.json")]:
    try:
        if p.stat().st_size < 10:  # tiny file likely empty/error
            st.sidebar.warning(f"{p.name} looks empty. Check API key/permissions and base URL.")
    except FileNotFoundError:
        st.sidebar.warning(f"{p.name} not found after fetch.")

# Now load from disk (cache-busted by file stats so prices refresh)
_stat = DATA.stat()
df = load_portfolio((_stat.st_mtime, _stat.st_size))

# --- DEBUG: show columns & a few rows so we can see what T212 returns
with st.sidebar.expander("Debug: portfolio columns", expanded=False):
    st.write(sorted(df.columns.tolist()))
    st.write(df.head(3))

# ---- Company / Instrument name (best effort, with optional instruments.json) ----
name_like_cols = [
    "name", "instrument.name", "displayName", "display_name",
    "company", "title", "instrumentName", "instrument_name", "security.name"
]
name_col = next((c for c in df.columns if c in name_like_cols), None)

if name_col:
    df["company"] = df[name_col].astype(str)
else:
    # Optional: merge from a local instruments export if you have it
    def _load_instrument_names(path=Path("data") / "instruments.json"):
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("items", raw)
        m = pd.json_normalize(items)
        tcol = next((c for c in m.columns if "ticker" in c.lower()), None)
        ncol = next((c for c in m.columns if "name"   in c.lower()), None)
        if tcol and ncol:
            m = m[[tcol, ncol]].rename(columns={tcol: "symbol", ncol: "company"})
            return m
        return None

    name_map = _load_instrument_names()

    # Small built-in mapping you can extend
    builtin_map = {
        "GOOGL_US_EQ": "Alphabet Inc. (Class A)",
        "MA_US_EQ":    "Mastercard Inc.",
        "SPXLI_EQ":    "S&P 500 UCITS ETF (Acc)",
        "GAWI_EQ":     "Gawain Plc",  # change to your actual LSE holding name
    }
    df["company"] = df["symbol"].map(builtin_map)

    if name_map is not None:
        df = df.merge(name_map, on="symbol", how="left", suffixes=("", "_from_map"))
        df["company"] = df["company"].combine_first(df["company_from_map"])
        if "company_from_map" in df.columns:
            df.drop(columns=["company_from_map"], inplace=True)

    # Final tidy fallback from ticker if still missing
    def _pretty(sym: str) -> str:
        s = str(sym)
        for tag in ["_US_EQ", "_EQ", "_US", "_GBX", "_GB"]:
            s = s.replace(tag, "")
        return s.replace("_", " ").strip()
    df["company"] = df["company"].fillna(df["symbol"].apply(_pretty))

@st.cache_data
def load_true_avg_cost(path: Path = Path("data") / "transactions.json"):
    """
    Compute true GBP average cost per ticker from T212 transactions history.
    Formula: sum(GBP spent on BUYs incl. fees & stamp) / sum(shares bought).
    Accepts either {"items":[...]} or a raw list.
    Returns (df, err) where df has: symbol, shares_bought, total_spend_gbp, true_avg_cost_gbp
    """
    if not path.exists():
        return None, "File not found"

    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items", raw)
    if not isinstance(items, list) or len(items) == 0:
        return None, "No items in transactions.json"

    t = pd.json_normalize(items)

    # Try to auto-detect columns
    action_col  = next((c for c in t.columns if c.lower() in {"action", "type"}), None)
    ticker_col  = next((c for c in t.columns if "ticker" in c.lower()), None)
    qty_col     = next((c for c in t.columns if "quantity" in c.lower() or "shares" in c.lower()), None)
    total_col   = next((c for c in t.columns if "totalamount" in c.replace("_", "").lower() or c.lower() == "total"), None)
    charge_col  = next((c for c in t.columns if "charge" in c.lower()), None)
    stamp_col   = next((c for c in t.columns if "stamp" in c.lower()), None)

    if any(c is None for c in [action_col, ticker_col, qty_col, total_col]):
        return None, f"Unexpected schema. Columns: {list(t.columns)}"

    # Coerce numeric
    for c in [qty_col, total_col, charge_col, stamp_col]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    if charge_col not in t.columns: t[charge_col] = 0.0
    if stamp_col  not in t.columns: t[stamp_col]  = 0.0

    # Keep BUYs
    buys = t[t[action_col].astype(str).str.contains("buy", case=False, na=False)].copy()
    if buys.empty:
        return None, "No buy transactions found."

    # T212 total amount is in ACCOUNT CURRENCY (GBP) → treat as GBP and add fees/stamp
    buys["spend_gbp"] = buys[total_col].fillna(0) + buys[charge_col].fillna(0) + buys[stamp_col].fillna(0)

    agg = (
        buys.groupby(buys[ticker_col])
            .agg(shares_bought=(qty_col, "sum"), total_spend_gbp=("spend_gbp", "sum"))
            .reset_index()
    )
    agg.rename(columns={ticker_col: "symbol"}, inplace=True)
    agg["true_avg_cost_gbp"] = agg["total_spend_gbp"] / agg["shares_bought"]
    return agg, None

@st.cache_data(ttl=6 * 3600)  # cache for 6 hours
def get_usd_gbp_rate():
    """
    Returns (rate, source, fetched_at) or (None, None, None) on failure.
    Tries two free endpoints: frankfurter.app (ECB) then exchangerate.host.
    """
    # Provider 1: frankfurter.app (ECB)
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "GBP"},
            timeout=8,
        )
        r.raise_for_status()
        rate = float(r.json()["rates"]["GBP"])
        return rate, "frankfurter.app (ECB)", datetime.utcnow().isoformat() + "Z"
    except Exception:
        pass

    # Provider 2: exchangerate.host
    try:
        r = requests.get(
            "https://api.exchangerate.host/latest",
            params={"base": "USD", "symbols": "GBP"},
            timeout=8,
        )
        r.raise_for_status()
        rate = float(r.json()["rates"]["GBP"])
        return rate, "exchangerate.host", datetime.utcnow().isoformat() + "Z"
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
    # Optional: allow override if you want to test scenarios
    if st.sidebar.toggle("Override FX rate", value=False, key="fx_override"):
        usd_to_gbp = st.sidebar.number_input(
            "USD → GBP (override)", value=float(f"{auto_rate:.4f}"), min_value=0.5, max_value=1.5, step=0.01
        )
    else:
        usd_to_gbp = auto_rate

with st.sidebar.expander("Day change settings", expanded=False):
    if st.button("Reset today’s open prices"):
        # wipe only today's bucket
        _store = _load_open_prices()
        _store[_anchor_date_iso()] = {}
        _save_open_prices(_store)
        st.success("Today's open prices reset. Reload to re-anchor.")

# Convert to GBP:
# - London listings in pence (GBX) already handled earlier (price_gbp_guess).
# - US listings are marked by "_US_" in the T212 symbol (e.g., GOOGL_US_EQ, MA_US_EQ).
def gbp_price(row):
    p = row["price_raw"]
    sym = str(row["symbol"])
    if p is None:
        return None
    if "_US_" in sym:
        return float(p) * usd_to_gbp       # USD → GBP
    return float(row["price_gbp_guess"])   # GBX already scaled to GBP

df["price_gbp"] = df.apply(gbp_price, axis=1)
df["market_value_gbp"] = df["shares"] * df["price_gbp"]

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Positions", f"{len(df):,}")
c2.metric("Total Shares", f"{int(df['shares'].sum()):,}")
c3.metric("Market Value (GBP)", f"£{df['market_value_gbp'].sum():,.0f}")

st.caption("GBX handled automatically; USD converted via the sidebar rate.")

# =======================
# Dividends loader (T212 schema: {items:[...], nextPagePath})
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

    # Keep only dividend cash events; drop interest/adjustments
    if "type" in df_div.columns:
        allowed = {"DIVIDEND", "DIVIDEND_CASH", "CASH_DIVIDEND"}
        df_div = df_div[df_div["type"].astype(str).str.upper().isin(allowed)]

    # Exclude interest-like rows
    if "reference" in df_div.columns:
        df_div = df_div[~df_div["reference"].astype(str).str.contains("interest", case=False, na=False)]

    # Expect paidOn + amount
    if "paidOn" not in df_div.columns or "amount" not in df_div.columns:
        return None, f"Unexpected columns: {list(df_div.columns)}"

    # Robust date parse
    paid = df_div["paidOn"].astype(str).str.strip().str.replace("Z", "", regex=False)
    dt = pd.to_datetime(paid, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(paid, format="%Y-%m-%d", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(paid, unit="ms", errors="coerce")

    df_div["dt"] = dt
    if df_div["dt"].isna().all():
        return None, "Could not parse 'paidOn' dates."

    out = (
        pd.DataFrame({
            "dt": df_div["dt"],
            "ticker": df_div.get("ticker"),
            "amount_gbp": pd.to_numeric(df_div["amount"], errors="coerce"),  # Account currency is GBP
        })
        .dropna(subset=["dt", "amount_gbp"])
        .sort_values("dt")
    )
    return out, None

# ---------- Holdings: richer table (native prices, GBP values, returns) ----------
st.subheader("Holdings")

# 1) Avg cost per share from payload (best-effort, still native)
avg_candidates = [c for c in df.columns if "average" in c.lower() and "price" in c.lower()]
df["avg_cost_raw"] = pd.to_numeric(df[avg_candidates[0]], errors="coerce") if avg_candidates else None

# 2) Listing currency from symbol
def _ccy(sym: str) -> str:
    """Infer listing ccy from T212 symbol convention."""
    return "USD" if "_US_" in str(sym) else "GBP"
df["ccy"] = df["symbol"].apply(_ccy)

# 3) Native price & native avg cost (no FX)
# Ensure avg cost is numeric
df["avg_cost_raw"] = pd.to_numeric(df["avg_cost_raw"], errors="coerce")

def _price_native(row):
    p = row["price_raw"]
    if pd.isna(p):
        return np.nan
    if row["ccy"] == "USD":
        return float(p)  # USD already native
    # LSE pence → GBP if huge
    return float(p) / 100.0 if p > 1000 else float(p)

def _avg_native(row):
    x = row["avg_cost_raw"]
    if pd.isna(x):
        return np.nan
    if row["ccy"] == "USD":
        return float(x)
    return float(x) / 100.0 if x > 1000 else float(x)

df["price_native"] = df.apply(_price_native, axis=1)
df["avg_cost_native"] = df.apply(_avg_native, axis=1)

# ---- Allow manual mapping of "Day Change" fields if present in payload ----
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

# --- Dividends per ticker (GBP) ---
div_tbl, _div_err = load_dividends_t212(Path("data") / "dividends.json")
if _div_err is None and div_tbl is not None:
    by_ticker = (
        div_tbl.groupby("ticker", as_index=False)["amount_gbp"].sum()
               .rename(columns={"ticker": "symbol", "amount_gbp": "dividends_gbp"})
    )
    df = df.merge(by_ticker, on="symbol", how="left")
else:
    df["dividends_gbp"] = None

# --- TRUE GBP cost basis from transactions (buys only) ---
true_costs, tc_err = load_true_avg_cost(Path("data") / "transactions.json")
if tc_err is None and true_costs is not None:
    df = df.merge(true_costs[["symbol", "true_avg_cost_gbp"]], on="symbol", how="left")
    df["cost_basis_gbp"] = df["true_avg_cost_gbp"] * df["shares"]
else:
    df["true_avg_cost_gbp"] = None
    df["cost_basis_gbp"] = None

# --- Portfolio value in GBP (already computed earlier) ---
df["total_value_gbp"] = pd.to_numeric(df["market_value_gbp"], errors="coerce")
portfolio_total = float(df["total_value_gbp"].sum())
df["weight_pct"] = np.where(
    portfolio_total > 0,
    (df["total_value_gbp"] / portfolio_total) * 100.0,
    0.0,
)

# --- Day change calculation ---
# Strategy:
# 1) If API provides a previous close column, use it (native).
# 2) Else, use persisted "today open" prices captured on first run of the day.
# 3) Optional manual mapping (sidebar) can override.

_prev_close_candidates = [
    "previousClose","prevClose","lastClose","closePrevious",
    "pricePrevClose","previous_close","priorClose"
]
prev_col = next((c for c in _prev_close_candidates if c in df.columns), None)

df["day_change_gbp"] = float("nan")
df["day_change_pct"] = float("nan")

if prev_col:
    # API-driven day change (native)
    def _prev_native(row):
        v = row[prev_col]
        if v is None or pd.isna(v): return None
        v = float(v)
        return v if row["ccy"] == "USD" else (v / 100.0 if v > 1000 else v)
    df["prev_close_native"] = df.apply(_prev_native, axis=1)
    df["day_change_native"] = df["price_native"] - df["prev_close_native"]
else:
    # Persisted "today open" prices (native)
    store, today_key = ensure_today_open_prices(df)
    today_opens = store.get(today_key, {})
    def _open_native(row):
        p = today_opens.get(str(row["symbol"]))
        return float(p) if p is not None else float("nan")
    df["prev_close_native"] = df.apply(_open_native, axis=1)  # reuse column name
    df["day_change_native"] = df["price_native"] - df["prev_close_native"]

# Absolute £ day change (convert USD rows with FX and scale by shares)
def _native_to_gbp(row, v):
    if pd.isna(v): return float("nan")
    return float(v) * (usd_to_gbp if row["ccy"] == "USD" else 1.0)

df["day_change_gbp"] = df.apply(
    lambda r: r["shares"] * _native_to_gbp(r, r["day_change_native"]),
    axis=1
)

# % day change (native %, independent of FX)
df["day_change_pct"] = (df["price_native"] / df["prev_close_native"] - 1.0) * 100

# ---- Manual mapping (sidebar) overrides if user selected columns ----
if 'sel_abs' in locals() and sel_abs != "<auto>" and sel_abs in df.columns:
    tmp_abs = pd.to_numeric(df[sel_abs], errors="coerce")

    # Heuristic: detect per-share vs position-level
    try:
        med_price = pd.to_numeric(df["price_native"], errors="coerce").median()
        med_abs   = tmp_abs.abs().median()
        per_share = bool(med_price and med_abs and med_abs < (3 * med_price))
    except Exception:
        per_share = False

    def _to_gbp(row, val):
        if pd.isna(val): return float("nan")
        scaled = (val * row["shares"]) if per_share else val
        return float(scaled) * (usd_to_gbp if row["ccy"] == "USD" else 1.0)

    df["day_change_gbp"] = df.apply(lambda r: _to_gbp(r, tmp_abs.loc[r.name]), axis=1)

if 'sel_pct' in locals() and sel_pct != "<auto>" and sel_pct in df.columns:
    df["day_change_pct"] = pd.to_numeric(df[sel_pct], errors="coerce")

# --- Normalise day-change outputs (ensure numeric, fill NaN→0 for display)
df["day_change_gbp"] = pd.to_numeric(df["day_change_gbp"], errors="coerce")
df["day_change_pct"] = pd.to_numeric(df["day_change_pct"], errors="coerce")

def _gbp_from_native(row, x):
    if pd.isna(x): 
        return float("nan")
    return float(x) * (usd_to_gbp if row["ccy"] == "USD" else 1.0)

# 8) Capital gains (GBP) & total return (GBP / %)
# Ensure true_avg_cost_gbp & dividends_gbp are numeric
df["true_avg_cost_gbp"] = pd.to_numeric(df["true_avg_cost_gbp"], errors="coerce")
df["dividends_gbp"] = pd.to_numeric(df["dividends_gbp"], errors="coerce")

# True cost (GBP) from transactions when available
df["cost_basis_gbp"] = df["shares"] * df["true_avg_cost_gbp"]

# Fallback: FX-adjusted native average
approx_cost_each_gbp = df.apply(lambda r: _gbp_from_native(r, r["avg_cost_native"]), axis=1)
df["cost_basis_gbp_approx"] = df["shares"] * approx_cost_each_gbp

# Prefer true; else approx
df["cost_basis_gbp_effective"] = df["cost_basis_gbp"].combine_first(df["cost_basis_gbp_approx"])

# Gains / returns (numeric & safe)
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

# 9) Native P/L % (price vs avg in native currency, FX-free)
df["pl_pct_native"] = (
    (df["price_native"] - df["avg_cost_native"]) / df["avg_cost_native"] * 100
)

# 10) Build the display table

# Clean display numerics (ensures formatting works uniformly)
num_cols = [
    "price_native",
    "avg_cost_native",
    "day_change_gbp",
    "day_change_pct",
    "total_return_gbp",
    "total_return_pct",
    "dividends_gbp",
    "capital_gains_gbp",
    "total_value_gbp",
    "weight_pct",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

view = df[[
    "company",
    "shares",
    "ccy",
    "price_native",
    "avg_cost_native",
    "day_change_gbp",
    "day_change_pct",
    "total_return_gbp",
    "total_return_pct",
    "dividends_gbp",
    "capital_gains_gbp",
    "total_value_gbp",   # <— add this
    "weight_pct",
]].copy()

st.dataframe(
    view.sort_values("total_value_gbp", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={
        "company": "Name",
        "shares": st.column_config.NumberColumn("No. of Shares", format="%.0f"),
        "ccy": st.column_config.TextColumn("Ccy"),
        "price_native":    st.column_config.NumberColumn("Live Share Price",  format="%.2f"),
        "avg_cost_native": st.column_config.NumberColumn("Avg. Share Price",  format="%.2f"),
        "day_change_gbp":  st.column_config.NumberColumn("Day Change £",      format="£%.0f"),
        "day_change_pct":  st.column_config.NumberColumn("Day Change %",      format="%.2f%%"),
        "total_return_gbp":st.column_config.NumberColumn("Total Return £",    format="£%.0f"),
        "total_return_pct":st.column_config.NumberColumn("Total Return %",    format="%.2f%%"),
        "dividends_gbp":   st.column_config.NumberColumn("Dividends",         format="£%.0f"),
        "capital_gains_gbp": st.column_config.NumberColumn("Capital Gains",   format="£%.0f"),
        "total_value_gbp": st.column_config.NumberColumn("Holding Value", format="£%.0f"),
        "weight_pct":      st.column_config.NumberColumn("Weight %",          format="%.2f%%"),
    },
)

# Helpful notes for what’s estimated/missing
notes = []
if df["true_avg_cost_gbp"].isna().any():
    notes.append("Using **approximate cost basis** from average price for some rows (FX-adjusted). Upload transactions.json for true GBP cost.")
if df["day_change_gbp"].isna().all():
    notes.append("Day change anchored to today’s first seen price (or map fields in the sidebar if your API provides them).")
if notes:
    st.caption(" • ".join(notes))

# ---- Render dividends timeline ----
from pandas.api.types import is_datetime64_any_dtype as is_datetime

div, err = load_dividends_t212(DIV_FILE)

st.subheader("Dividends timeline")
if err:
    st.write("Dividends (inspect)")
    st.warning(err)
else:
    # Ensure datetime and clean rows
    if not is_datetime(div["dt"]):
        div["dt"] = pd.to_datetime(div["dt"], errors="coerce")
    div = div.dropna(subset=["dt", "amount_gbp"]).copy()
    div["amount_gbp"] = pd.to_numeric(div["amount_gbp"], errors="coerce")
    div = div.dropna(subset=["amount_gbp"])

    # --- Year filter (populate sidebar with real years)
    years = sorted(div["dt"].dt.year.unique().tolist())
    # Update the placeholder selectbox made earlier
    selected_year = st.sidebar.selectbox("Year", options=["All"] + years, index=0)

    if selected_year != "All":
        div = div[div["dt"].dt.year == int(selected_year)]

    # --- Monthly aggregation with year included (YYYY-MM)
    div["year_month"] = div["dt"].dt.to_period("M")
    monthly = div.groupby("year_month", as_index=False)["amount_gbp"].sum()
    monthly["year_month"] = monthly["year_month"].astype(str)

    st.bar_chart(monthly, x="year_month", y="amount_gbp", use_container_width=True)
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
# Backend performance plumbing (no UI yet)
# =======================
from jobs.snapshot import append_today_snapshot_if_missing
from pdperf.series import read_nav, daily_returns_twr, cumulative_return, cagr
from pdperf.cashflows import build_cash_flows
from bench.sp500 import get_sp500_daily

# 1) Persist today's NAV snapshot (creates/updates data/nav_daily.csv)
try:
    append_today_snapshot_if_missing(df)
except Exception as e:
    st.sidebar.warning(f"NAV snapshot not updated: {e}")

# 2) (Optional debug) Compute perf vs S&P 500 in GBP (sidebar captions)
try:
    # Align benchmark to your actual NAV history window
    today_key = _anchor_date_iso()

    nav = read_nav()  # <- Series indexed by date (not a DataFrame)
    flows = build_cash_flows(Path("data") / "transactions.json")
    port_daily = daily_returns_twr(nav, flows)   # DataFrame: [date, r_port]

    # Start = earliest NAV index (since read_nav() returns a Series)
    perf_start = pd.to_datetime(nav.index).min().strftime("%Y-%m-%d")

    # Fetch S&P over same window (GBP returns)
    sp = get_sp500_daily(perf_start, today_key)  # DataFrame: [date, daily_ret]

    # Make both sides the same type for joining/filters
    port_daily["date"] = pd.to_datetime(port_daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sp["date"]         = pd.to_datetime(sp["date"],         errors="coerce").dt.strftime("%Y-%m-%d")

    # Align on common dates
    merged = (
        port_daily.merge(
            sp[["date", "daily_ret"]].rename(columns={"daily_ret": "r_bench"}),
            on="date",
            how="inner"
        )
        .dropna(subset=["r_port", "r_bench"])
        .sort_values("date")
    )

    # Since-start cumulative
    since_start_port = cumulative_return(merged, perf_start, today_key)
    since_start_bench = float((1 + merged["r_bench"]).prod() - 1) if not merged.empty else float("nan")

    # YTD slice (kept for quick scan)
    year_start = f"{datetime.utcnow().year}-01-01"
    ytd_slice = merged[(merged["date"] >= year_start) & (merged["date"] <= today_key)]
    ytd_port  = float((1 + ytd_slice["r_port"]).prod()  - 1) if not ytd_slice.empty else float("nan")
    ytd_bench = float((1 + ytd_slice["r_bench"]).prod() - 1) if not ytd_slice.empty else float("nan")

    if pd.notna(since_start_port) and pd.notna(since_start_bench):
        st.sidebar.caption(f"Since {perf_start}: Portfolio {since_start_port:.2%} vs S&P {since_start_bench:.2%}")
    if pd.notna(ytd_port) and pd.notna(ytd_bench):
        st.sidebar.caption(f"YTD (backend): Portfolio {ytd_port:.2%} vs S&P {ytd_bench:.2%}")

    # === NAV vs S&P 500 (rebased to 100 at the strategy anchor) ===
    anchor = "2025-01-01"  # strategy change date

    plot = merged[merged["date"] >= anchor].copy()
    if plot.empty:
        st.info("Not enough data after the anchor date to draw the NAV vs S&P chart.")
    else:
        # Cumulative indices: 100 * Π(1 + daily return)
        plot["Portfolio (TWR)"] = (100.0 * (1.0 + plot["r_port"]).cumprod()).astype("float64")
        plot["S&P 500 (GBP)"]   = (100.0 * (1.0 + plot["r_bench"]).cumprod()).astype("float64")

        st.subheader("NAV vs S&P 500 (rebased to 100)")
        # Build long-form DF for Altair
        plot_alt = plot.copy()
        plot_alt["date"] = pd.to_datetime(plot_alt["date"], errors="coerce")
        plot_alt = plot_alt.melt(
            id_vars=["date"],
            value_vars=["Portfolio (TWR)", "S&P 500 (GBP)"],
            var_name="series",
            value_name="index"
        )

        # Add 5% headroom/footroom so the lines can breathe
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

        st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.sidebar.info(f"Perf debug unavailable: {e}")