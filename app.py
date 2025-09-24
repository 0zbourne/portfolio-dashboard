import json, pandas as pd, streamlit as st
from pathlib import Path

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

# ---- Sidebar: FX ----
st.sidebar.header("Settings")
usd_to_gbp = st.sidebar.number_input("USD → GBP rate", value=0.78, min_value=0.5, max_value=1.5, step=0.01)

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

# Table
st.subheader("Holdings")
st.dataframe(
    df[["symbol","shares","price_gbp","market_value_gbp"]]
      .sort_values("market_value_gbp", ascending=False),
    use_container_width=True,
)

# Quick weights chart
st.subheader("Weights by Holding")
weights = df[["symbol","market_value_gbp"]].sort_values("market_value_gbp", ascending=False)
weights["weight_%"] = (weights["market_value_gbp"] / weights["market_value_gbp"].sum()) * 100
st.bar_chart(weights, x="symbol", y="market_value_gbp", use_container_width=True)


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
