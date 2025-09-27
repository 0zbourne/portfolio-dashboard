import json, pandas as pd, streamlit as st
from pathlib import Path

import requests
from datetime import datetime

import os

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

@st.cache_data
def load_portfolio():
    with open(DATA, "r", encoding="utf-8") as f:
        items = json.load(f)
    df = pd.DataFrame(items)

    # Minimal derived columns
    df.rename(columns={"ticker":"symbol","quantity":"shares","currentPrice":"price_raw"}, inplace=True)

    # Very rough GBP conversion for LSE tickers quoted in pence (GBX):
    # If price looks huge (e.g., 14680 = 146.80 GBP), treat as pence.
    df["price_gbp_guess"] = df["price_raw"].apply(lambda x: x/100 if x and x>1000 else x)
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

# Now load from disk as before
df = load_portfolio()

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
    return "USD" if "_US_" in str(sym) else "GBP"
df["ccy"] = df["symbol"].apply(_ccy)

# 3) Native price & native avg cost (no FX)
def _price_native(row):
    p = row["price_raw"]
    if p is None or pd.isna(p):
        return None
    p = float(p)
    if row["ccy"] == "USD":
        return p                   # already USD
    return p/100.0 if p > 1000 else p   # GBX → GBP for UK listings

def _avg_native(row):
    x = row["avg_cost_raw"]
    if x is None or pd.isna(x):
        return None
    x = float(x)
    if row["ccy"] == "USD":
        return x                   # keep USD; do NOT guess FX
    return x/100.0 if x > 1000 else x   # GBX → GBP for UK listings

df["price_native"]    = df.apply(_price_native, axis=1)
df["avg_cost_native"] = df.apply(_avg_native, axis=1)

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
df["total_value_gbp"] = df["market_value_gbp"]
portfolio_total = df["total_value_gbp"].sum()
df["weight_pct"] = (df["total_value_gbp"] / portfolio_total) * 100

# --- Day change (try prev close first; then wide-net fallbacks) ---
_prev_close_candidates = [
    "previousClose","prevClose","lastClose","closePrevious",
    "pricePrevClose","previous_close","priorClose"
]
prev_col = next((c for c in _prev_close_candidates if c in df.columns), None)

df["day_change_gbp"] = float("nan")
df["day_change_pct"] = float("nan")

if prev_col:
    def _prev_native(row):
        v = row[prev_col]
        if v is None or pd.isna(v): return None
        v = float(v)
        return v if row["ccy"] == "USD" else (v / 100.0 if v > 1000 else v)

    df["prev_close_native"] = df.apply(_prev_native, axis=1)
    df["day_change_native"] = df["price_native"] - df["prev_close_native"]

    # absolute £ day change
    df["day_change_gbp"] = df.apply(
        lambda r: r["shares"] * (r["day_change_native"] * (usd_to_gbp if r["ccy"] == "USD" else 1.0))
        if pd.notna(r["day_change_native"]) else float("nan"),
        axis=1
    )
    # % day change
    df["day_change_pct"] = (df["price_native"] / df["prev_close_native"] - 1.0) * 100

# ---- Fallbacks if the API didn't give us a previous close ----
def _pick(colnames, must_have_any, also_any=None):
    """Pick first column whose name contains ALL tokens in `must_have_any`
       and (if provided) ANY token in `also_any` (case-insensitive)."""
    low = {c: c.lower() for c in colnames}
    for c, lc in low.items():
        if all(tok in lc for tok in must_have_any) and (not also_any or any(tok in lc for tok in also_any)):
            return c
    return None

if df["day_change_gbp"].isna().all():
    # absolute “today/day” PnL or change
    abs_col = _pick(df.columns, ["day"], ["pnl","pl","change","chg","diff","ret"]) \
          or  _pick(df.columns, ["today"], ["pnl","pl","change","chg","diff","ret"])
    if abs_col:
        vals = pd.to_numeric(df[abs_col], errors="coerce")
        df["day_change_gbp"] = df.apply(
            lambda r: (vals.loc[r.name] * (usd_to_gbp if r["ccy"] == "USD" else 1.0))
            if pd.notna(vals.loc[r.name]) else float("nan"),
            axis=1
        )

    # percentage “today/day”
    pct_col = _pick(df.columns, ["day"],   ["pct","percent","%","ret"]) \
          or  _pick(df.columns, ["today"], ["pct","percent","%","ret"])
    if pct_col:
        df["day_change_pct"] = pd.to_numeric(df[pct_col], errors="coerce")

def _gbp_from_native(row, x):
    if pd.isna(x): 
        return float("nan")
    return float(x) * (usd_to_gbp if row["ccy"] == "USD" else 1.0)

# 8) Capital gains (GBP) & total return (GBP / %)
# Preferred: true GBP cost from transactions.json (already GBP)
df["cost_basis_gbp"] = df["shares"] * df["true_avg_cost_gbp"]

# Fallback: approximate cost from avg_cost_native (FX-adjust USD rows)
approx_cost_each_gbp = df.apply(lambda r: _gbp_from_native(r, r["avg_cost_native"]), axis=1)
df["cost_basis_gbp_approx"] = df["shares"] * approx_cost_each_gbp

# Use true when available; otherwise approx
df["cost_basis_gbp_effective"] = df["cost_basis_gbp"].combine_first(df["cost_basis_gbp_approx"])

# Gains & returns
df["capital_gains_gbp"] = df["total_value_gbp"] - df["cost_basis_gbp_effective"]
df["total_return_gbp"]  = df["capital_gains_gbp"] + df["dividends_gbp"]
df["total_return_pct"]  = (df["total_return_gbp"] / df["cost_basis_gbp_effective"]) * 100

# 9) Native P/L % (price vs avg in native currency, FX-free)
df["pl_pct_native"] = (
    (df["price_native"] - df["avg_cost_native"]) / df["avg_cost_native"] * 100
)

# 10) Build the display table
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
    notes.append("Could not find day-change fields in the payload; columns show ‘—’.")
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
