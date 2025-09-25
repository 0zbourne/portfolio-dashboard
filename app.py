import json, pandas as pd, streamlit as st
from pathlib import Path

import requests
from datetime import datetime

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

df = load_portfolio()

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

# ---------- Holdings: show prices & avg cost in native currency; GBP only for portfolio value ----------
st.subheader("Holdings")

# 1) Find avg cost per share from T212 payload (best-effort)
avg_candidates = [c for c in df.columns if "average" in c.lower() and "price" in c.lower()]
df["avg_cost_raw"] = pd.to_numeric(df[avg_candidates[0]], errors="coerce") if avg_candidates else None

# 2) Listing currency from symbol
def _ccy(sym: str) -> str:
    return "USD" if "_US_" in str(sym) else "GBP"

df["ccy"] = df["symbol"].apply(_ccy)

# 3) Native price & native avg cost (don’t FX-convert here)
def _price_native(row):
    p = row["price_raw"]
    if p is None or pd.isna(p): return None
    p = float(p)
    if row["ccy"] == "USD":
        return p                    # already USD
    return p/100.0 if p > 1000 else p   # GBX → GBP for UK tickers

def _avg_native(row):
    x = row["avg_cost_raw"]
    if x is None or pd.isna(x): return None
    x = float(x)
    if row["ccy"] == "USD":
        return x                    # keep USD; do NOT guess FX
    return x/100.0 if x > 1000 else x   # GBX → GBP for UK tickers

df["price_native"]    = df.apply(_price_native, axis=1)
df["avg_cost_native"] = df.apply(_avg_native, axis=1)

# 4) GBP portfolio value (we already computed market_value_gbp earlier)
df["total_value_gbp"] = df["market_value_gbp"]
portfolio_total = df["total_value_gbp"].sum()
df["weight_pct"] = (df["total_value_gbp"] / portfolio_total) * 100

# 5) Pretty display strings (native currency)
def _fmt(val, ccy):
    if val is None or pd.isna(val): return ""
    return f"${val:,.2f}" if ccy == "USD" else f"£{val:,.2f}"

df["price_display"]    = df.apply(lambda r: _fmt(r["price_native"],    r["ccy"]), axis=1)
df["avg_cost_display"] = df.apply(lambda r: _fmt(r["avg_cost_native"], r["ccy"]), axis=1)

# 6) Table (no P/L until we wire true GBP cost from transactions)
view = df[[
    "symbol",
    "shares",
    "price_display",       # native
    "avg_cost_display",    # native
    "total_value_gbp",     # GBP
    "weight_pct",
]].copy()

st.dataframe(
    view.sort_values("total_value_gbp", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={
        "symbol": "Ticker",
        "shares": st.column_config.NumberColumn("Shares", format="%.0f"),
        "price_display": st.column_config.TextColumn("Price (native)"),
        "avg_cost_display": st.column_config.TextColumn("Avg Cost (native)"),
        "total_value_gbp": st.column_config.NumberColumn("Total Value (GBP)", format="£%.0f"),
        "weight_pct": st.column_config.NumberColumn("Weight %", format="%.2f%%"),
    },
)

st.caption(
    "Prices & average costs are shown in each instrument’s native currency. "
    "Portfolio values are converted to GBP using the FX rate on the left. "
    "P/L is hidden until we import BUY history to compute the true GBP cost basis."
)

# =======================
# Dividends timeline (T212 schema: {items:[...], nextPagePath})
# =======================
DIV_FILE = Path("data") / "dividends.json"

@st.cache_data
def load_dividends_t212(path=DIV_FILE):
    if not path.exists():
        return None, "File not found"

    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items", [])
    if not isinstance(items, list) or len(items) == 0:
        return None, "No items in dividends.json"

    df = pd.json_normalize(items) 

    # Keep only actual dividend cash events; drop interest/adjustments
    if "type" in df.columns:
        allowed = {"DIVIDEND", "DIVIDEND_CASH", "CASH_DIVIDEND"}
        df = df[df["type"].astype(str).str.upper().isin(allowed)]

    # Some feeds label interest in 'reference'—exclude those too
    if "reference" in df.columns:
        df = df[~df["reference"].astype(str).str.contains("interest", case=False, na=False)]

    # Drop rows with no paidOn entirely – they create the NaT bucket later
    df = df[ df["paidOn"].notna() ]
    
    # Expect: ['ticker','reference','quantity','amount','grossAmountPerShare','amountInEuro','paidOn','type']
    if "paidOn" not in df.columns or "amount" not in df.columns:
        return None, f"Unexpected columns: {list(df.columns)}"

    # --- robust date parse (ALL INSIDE the function) ---
    paid = df["paidOn"].astype(str).str.strip().str.replace("Z", "", regex=False)
    dt = pd.to_datetime(paid, errors="coerce")                      # try ISO
    if dt.isna().all():
        dt = pd.to_datetime(paid, format="%Y-%m-%d", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(paid, unit="ms", errors="coerce")

    df["dt"] = dt
    if df["dt"].isna().all():
        return None, f"Could not parse 'paidOn'. Examples: {paid.head(3).tolist()}"

    out = (
        pd.DataFrame({
            "dt": df["dt"],
            "ticker": df.get("ticker"),
            "amount_gbp": pd.to_numeric(df["amount"], errors="coerce"),  # T212 posts in account currency (GBP)
        })
        .dropna(subset=["dt", "amount_gbp"])
        .sort_values("dt")
    )
    return out, None

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
