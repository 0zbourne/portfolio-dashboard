# jobs/backfill.py
"""
Trading212 Order History → NAV Backfill with yfinance Prices
Includes debug export for granular NAV breakdown analysis.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, date
import os, json, math, time, traceback
import pandas as pd
import requests
import numpy as np

# --- Pandas configuration: force NumPy-native dtypes ---
try:
    pd.options.mode.dtype_backend = "numpy"
except Exception:
    pass
try:
    pd.options.mode.string_storage = "python"
except Exception:
    pass

# Optional dependency
try:
    import yfinance as yf
except Exception:
    yf = None

# ---------------------------
# Configuration
# ---------------------------
API_BASE = os.getenv("T212_API_BASE", "https://live.trading212.com")
API_KEY = os.getenv("T212_API_KEY")
API_SECRET = os.getenv("T212_API_SECRET")

DATA_DIR = Path("data")
NAV_CSV = DATA_DIR / "nav_daily.csv"
REPORT = DATA_DIR / "backfill_report.json"
OVERRIDES_PATH = DATA_DIR / "ticker_overrides.json"
DEBUG_CSV = DATA_DIR / "nav_debug_breakdown.csv"
SUMMARY_CSV = DATA_DIR / "nav_daily_summary.csv"
MAPPING_JSON = DATA_DIR / "ticker_mapping_debug.json"
TRACE_PATH = DATA_DIR / "_backfill_trace.txt"

# ---------------------------
# Authentication
# ---------------------------
def _t212_headers() -> dict:
    """Trading212 auth: Basic base64(KEY:SECRET) or legacy Apikey."""
    key = (API_KEY or "").strip()
    secret = (API_SECRET or "").strip()

    if key and secret:
        import base64
        token = base64.b64encode(f"{key}:{secret}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}", "Accept": "application/json"}

    if key:
        if not key.lower().startswith("apikey "):
            key = f"Apikey {key}"
        return {"Authorization": key, "Accept": "application/json"}

    raise RuntimeError("Missing T212_API_KEY in environment.")


# ---------------------------
# API Helpers
# ---------------------------
def _paged_get(url: str) -> list:
    """Follow nextPagePath until exhausted. Returns list of items."""
    items = []
    next_url = url
    for _ in range(1000):
        r = requests.get(next_url, headers=_t212_headers(), timeout=20)
        r.raise_for_status()
        payload = r.json()
        chunk = payload.get("items", payload if isinstance(payload, list) else [])
        if isinstance(chunk, list):
            items.extend(chunk)
        next_path = payload.get("nextPagePath")
        if not next_path:
            break
        next_url = API_BASE.rstrip("/") + next_path
        time.sleep(2.1)
    return items


# ---------------------------
# Ticker Mapping
# ---------------------------
def _load_overrides() -> dict:
    """Load ticker overrides from JSON. Keys normalized to uppercase."""
    if OVERRIDES_PATH.exists():
        try:
            raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
            return {k.strip().upper(): v for k, v in raw.items()}
        except Exception:
            pass
    return {}


def _infer_yf_symbol(t212_ticker: str, overrides: dict) -> tuple[str | None, str]:
    """
    Map Trading212 ticker to Yahoo Finance symbol.
    Returns (yf_symbol_or_none, currency 'GBP'|'USD'|None)
    """
    t = (t212_ticker or "").strip().upper()

    # 1) Explicit override
    if t in overrides:
        v = overrides[t]
        if isinstance(v, dict):
            return v.get("yf"), v.get("ccy", "GBP")
        if isinstance(v, str):
            return v, "GBP"

    # 2) US listings
    if "_US_" in t:
        core = t.split("_")[0]
        return core, "USD"

    # 3) LSE listings - strip suffixes, handle trailing 'L'
    core = t.replace("_GBX", "").replace("_GB", "").replace("_EQ", "")
    core = core.split("_")[0]
    if core.endswith("L") and len(core) >= 4:
        core = core[:-1]

    if core and core.isalpha() and 1 <= len(core) <= 5:
        return f"{core}.L", "GBP"

    return None, None


# ---------------------------
# Position Timeseries
# ---------------------------
def _build_position_timeseries(orders: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """
    Build daily position quantities from orders.
    Returns DataFrame: index=dates, columns=tickers, values=shares held.
    """
    if orders.empty:
        idx = pd.date_range(start, end, freq="D").date
        return pd.DataFrame(index=idx, dtype="float64")

    orders = orders.copy()
    orders["filledAt"] = pd.to_datetime(orders["filledAt"], errors="coerce", utc=True).dt.date
    orders = orders.dropna(subset=["filledAt", "ticker", "filledQuantity"])

    side_str = orders.get("side", "BUY").astype(str).str.upper()
    sign = np.where(side_str.str.startswith("S"), -1.0, 1.0)
    qty = pd.to_numeric(orders["filledQuantity"], errors="coerce").astype("float64")
    orders["signed_qty"] = qty * sign

    daily = (orders.groupby(["filledAt", "ticker"], as_index=False)["signed_qty"]
             .sum().rename(columns={"filledAt": "date"}))

    idx = pd.date_range(start, end, freq="D").date
    tickers = sorted(daily["ticker"].unique().tolist())
    mat = pd.DataFrame(0.0, index=idx, columns=tickers, dtype="float64")

    for _, r in daily.iterrows():
        d = r["date"]
        tk = r["ticker"]
        q = float(r["signed_qty"])
        mat.loc[mat.index >= d, tk] += q

    mat = mat.loc[:, (mat != 0).any(axis=0)].astype("float64")
    return mat


# ---------------------------
# Price Downloads
# ---------------------------
def _download_fx_usd_gbp(start: date, end: date) -> pd.Series:
    """Fetch USD→GBP rates from Frankfurter API."""
    url = f"https://api.frankfurter.app/{start}..{end}"
    r = requests.get(url, params={"from": "USD", "to": "GBP"}, timeout=20)
    r.raise_for_status()
    data = r.json().get("rates", {})
    fx = pd.DataFrame.from_dict(data, orient="index").rename(columns={"GBP": "usd_gbp"})
    fx.index = pd.to_datetime(fx.index).date
    return fx["usd_gbp"]


def _download_prices_batch(symbols: list[str], start: date, end: date, cal_idx: list) -> pd.DataFrame:
    """Download prices from yfinance with rate-limit handling."""
    if not symbols or yf is None:
        return pd.DataFrame(index=cal_idx, dtype="float64")

    last_err = None
    for sleep_s in (0, 2, 5, 10, 20, 40):
        if sleep_s:
            time.sleep(sleep_s)
        try:
            df = yf.download(
                symbols,
                start=str(start),
                end=str(end + timedelta(days=1)),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = df["Close"] if isinstance(df, pd.DataFrame) and "Close" in df else df
            if isinstance(close, pd.Series):
                close = close.to_frame()
            if close is None or close.empty:
                continue
            close.index = pd.to_datetime(close.index).date
            for c in close.columns:
                close[c] = pd.to_numeric(close[c], errors="coerce").astype("float64")
            return close.reindex(cal_idx)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Yahoo download failed: {last_err}")


# ---------------------------
# Core: Fetch Raw Data
# ---------------------------
def _fetch_orders(end_date: date) -> pd.DataFrame:
    """Fetch and normalize Trading212 orders."""
    fetch_from = "1970-01-01"
    url = f"{API_BASE}/api/v0/equity/history/orders?from={fetch_from}&to={end_date}"
    items = _paged_get(url)
    if not items:
        raise RuntimeError("No order history returned from Trading212.")

    o = pd.json_normalize(items)

    # Status filter
    status_col = next((c for c in ["status", "order.status"] if c in o.columns), None)
    if status_col:
        o = o[o[status_col].astype(str).str.upper().eq("FILLED")]

    # Column detection
    time_col = next((c for c in ["fill.filledAt", "filledAt", "order.filledAt", "order.createdAt"] if c in o.columns), None)
    if time_col is None:
        time_col = next((c for c in o.columns if str(c).endswith(".filledAt") or str(c).endswith("filledAt")), None)
    if time_col is None:
        raise RuntimeError(f"Could not find fill timestamp. Columns: {list(o.columns)}")

    ticker_col = next((c for c in ["order.ticker", "order.instrument.ticker", "ticker"] if c in o.columns), None)
    if ticker_col is None:
        raise RuntimeError(f"Could not find ticker column. Columns: {list(o.columns)}")

    qty_col = next((c for c in ["fill.quantity", "order.filledQuantity", "filledQuantity"] if c in o.columns), None)
    if qty_col is None:
        raise RuntimeError(f"Could not find quantity column. Columns: {list(o.columns)}")

    side_col = next((c for c in ["order.side", "side"] if c in o.columns), None)

    return pd.DataFrame({
        "ticker": o[ticker_col],
        "filledQuantity": o[qty_col],
        "filledAt": o[time_col],
        "side": o[side_col] if side_col else "BUY",
    })


# ---------------------------
# Main: NAV Backfill
# ---------------------------
def backfill_nav_from_orders(start: str = "2025-01-01", end: str | None = None) -> Path:
    """Rebuild nav_daily.csv from Trading212 orders + yfinance prices."""

    def _dump_trace(stage: str, pos=None, prices=None):
        try:
            with open(TRACE_PATH, "w", encoding="utf-8") as f:
                f.write(f"[STAGE] {stage}\n\n[TRACEBACK]\n")
                f.write(traceback.format_exc())
                if pos is not None:
                    f.write(f"\n[positions.dtypes]\n{pos.dtypes}\n")
                if prices is not None:
                    f.write(f"\n[prices.dtypes]\n{prices.dtypes}\n")
        except Exception:
            pass

    try:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date() if end else datetime.utcnow().date()

        # 1) Orders
        orders = _fetch_orders(d1)

        # 2) Positions
        pos = _build_position_timeseries(orders, d0, d1)

        # 3) Mapping
        overrides = _load_overrides()
        mapping = {t: _infer_yf_symbol(t, overrides) for t in pos.columns}

        # 4) Prices
        full_idx = pd.date_range(d0, d1, freq="D").date
        gbp_syms = [t for t, (y, ccy) in mapping.items() if y and ccy == "GBP"]
        usd_syms = [t for t, (y, ccy) in mapping.items() if y and ccy == "USD"]

        prices = pd.DataFrame(index=full_idx, dtype="float64")
        missing = []

        # GBP stocks
        if gbp_syms:
            ysyms = [mapping[t][0] for t in gbp_syms]
            gbp_px = _download_prices_batch(ysyms, d0, d1, full_idx)
            for t in gbp_syms:
                ysym = mapping[t][0]
                ser = gbp_px.get(ysym)
                if ser is None or ser.dropna().empty:
                    missing.append(t)
                    continue
                med = float(ser.dropna().median()) if ser.notna().any() else None
                if med and med > 1000.0:
                    ser = ser / 100.0
                prices[t] = ser

        # USD stocks
        if usd_syms:
            ysyms = [mapping[t][0] for t in usd_syms]
            usd_px = _download_prices_batch(ysyms, d0, d1, full_idx)
            fx = _download_fx_usd_gbp(d0, d1).reindex(full_idx)
            for t in usd_syms:
                ysym = mapping[t][0]
                ser = usd_px.get(ysym)
                if ser is None or ser.dropna().empty:
                    missing.append(t)
                    continue
                prices[t] = ser * fx

        # 5) Align
        keep = [t for t in pos.columns if t in prices.columns]
        pos = pos[keep].reindex(full_idx).ffill().fillna(0.0)
        prices = prices[keep].reindex(full_idx).ffill()

        # 6) NAV
        nav_vals = np.nansum(pos.to_numpy() * prices.to_numpy(), axis=1)

        nav = pd.DataFrame({
            "date": pd.to_datetime(full_idx).strftime("%Y-%m-%d"),
            "nav_gbp": nav_vals.astype("float64")
        })
        NAV_CSV.parent.mkdir(parents=True, exist_ok=True)
        nav.to_csv(NAV_CSV, index=False)

        REPORT.write_text(json.dumps({"missing_symbols": missing, "mapped": mapping}, indent=2), encoding="utf-8")

        return NAV_CSV

    except Exception:
        _dump_trace("FAILED", pos=locals().get("pos"), prices=locals().get("prices"))
        raise


# ==================================================
# DEBUG EXPORT - ADD THIS FUNCTION
# ==================================================
def export_debug_breakdown(start: str = "2025-01-01", end: str | None = None) -> Path:
    """
    Export detailed NAV breakdown for debugging.
    
    Outputs:
      - data/nav_debug_breakdown.csv: Every position, every day with qty, price, fx, value
      - data/nav_daily_summary.csv: Daily NAV totals with changes
      - data/ticker_mapping_debug.json: Ticker-to-yfinance mapping
    
    Open in Excel/Sheets and filter by date to find spikes.
    """
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date() if end else datetime.utcnow().date()

    # 1) Fetch orders and build positions
    orders = _fetch_orders(d1)
    pos = _build_position_timeseries(orders, d0, d1)

    # 2) Build mapping
    overrides = _load_overrides()
    mapping = {t: _infer_yf_symbol(t, overrides) for t in pos.columns}

    # 3) Download raw prices (separate from FX conversion)
    full_idx = pd.date_range(d0, d1, freq="D").date
    gbp_syms = [t for t, (y, ccy) in mapping.items() if y and ccy == "GBP"]
    usd_syms = [t for t, (y, ccy) in mapping.items() if y and ccy == "USD"]

    raw_prices = pd.DataFrame(index=full_idx, dtype="float64")
    missing = []

    # GBP raw prices
    if gbp_syms and yf:
        ysyms = [mapping[t][0] for t in gbp_syms]
        gbp_px = _download_prices_batch(ysyms, d0, d1, full_idx)
        for t in gbp_syms:
            ysym = mapping[t][0]
            ser = gbp_px.get(ysym)
            if ser is None or ser.dropna().empty:
                missing.append(t)
                continue
            raw_prices[t] = ser

    # USD raw prices (no FX yet)
    if usd_syms and yf:
        ysyms = [mapping[t][0] for t in usd_syms]
        usd_px = _download_prices_batch(ysyms, d0, d1, full_idx)
        for t in usd_syms:
            ysym = mapping[t][0]
            ser = usd_px.get(ysym)
            if ser is None or ser.dropna().empty:
                missing.append(t)
                continue
            raw_prices[t] = ser

    # 4) FX rates
    fx = _download_fx_usd_gbp(d0, d1).reindex(full_idx) if usd_syms else pd.Series(index=full_idx, dtype="float64")

    # 5) Build long-format debug DataFrame
    rows = []

    for date_val in full_idx:
        for ticker in pos.columns:
            if ticker not in raw_prices.columns:
                continue

            qty = float(pos.loc[date_val, ticker]) if ticker in pos.columns else 0.0
            if qty == 0:
                continue

            ysym, ccy = mapping.get(ticker, (None, None))
            price_raw = float(raw_prices.loc[date_val, ticker]) if ticker in raw_prices.columns else np.nan

            # GBX→GBP conversion for UK stocks
            price_gbp_raw = price_raw
            if ccy == "GBP" and price_raw > 1000.0:
                price_gbp_raw = price_raw / 100.0

            # FX for USD stocks
            fx_rate = float(fx.loc[date_val]) if ccy == "USD" and date_val in fx.index and pd.notna(fx.loc[date_val]) else 1.0

            # Final GBP price
            price_gbp = price_gbp_raw * fx_rate if ccy == "USD" else price_gbp_raw

            # Position value
            pos_value = qty * price_gbp

            rows.append({
                "date": date_val,
                "ticker": ticker,
                "yf_symbol": ysym,
                "currency": ccy,
                "quantity": qty,
                "price_raw": price_raw,
                "price_gbp_raw": price_gbp_raw,
                "fx_rate_usd_gbp": fx_rate if ccy == "USD" else "",
                "price_gbp": price_gbp,
                "position_value_gbp": pos_value,
            })

    debug_df = pd.DataFrame(rows)

    # 6) Add daily NAV summary
    daily_nav = debug_df.groupby("date")["position_value_gbp"].sum().reset_index()
    daily_nav.columns = ["date", "nav_total_gbp"]
    daily_nav["nav_change_gbp"] = daily_nav["nav_total_gbp"].diff()
    daily_nav["nav_change_pct"] = daily_nav["nav_total_gbp"].pct_change() * 100

    # Merge daily totals back
    debug_df = debug_df.merge(daily_nav, on="date", how="left")

    # Sort
    debug_df = debug_df.sort_values(["date", "position_value_gbp"], ascending=[False, False])

    # 7) Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    debug_df.to_csv(DEBUG_CSV, index=False)
    daily_nav.to_csv(SUMMARY_CSV, index=False)
    MAPPING_JSON.write_text(json.dumps(mapping, indent=2, default=str), encoding="utf-8")

    print(f"[DEBUG] Exported {len(debug_df)} rows to {DEBUG_CSV}")
    print(f"[DEBUG] Daily summary: {SUMMARY_CSV}")
    print(f"[DEBUG] Ticker mapping: {MAPPING_JSON}")

    return DEBUG_CSV
