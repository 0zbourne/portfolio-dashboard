# jobs/backfill.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, date
import os, json, math, time
import pandas as pd
import requests
import numpy as np

# Optional dependency; keep it local to this module
try:
    import yfinance as yf
except Exception:
    yf = None

API_BASE = os.getenv("T212_API_BASE", "https://live.trading212.com")
API_KEY  = os.getenv("T212_API_KEY")

DATA_DIR = Path("data")
NAV_CSV  = DATA_DIR / "nav_daily.csv"
REPORT   = DATA_DIR / "backfill_report.json"
OVERRIDES_PATH = DATA_DIR / "ticker_overrides.json"  # optional mapping file

# ---------------------------
# Helpers: fetch & pagination
# ---------------------------
def _auth_headers():
    if not API_KEY:
        raise RuntimeError("Missing T212_API_KEY in environment.")
    # T212 expects "Authorization: <apiKey>"
    return {"Authorization": API_KEY, "Accept": "application/json"}

def _paged_get(url: str):
    """Follow nextPagePath until exhausted. Returns list of items."""
    items = []
    next_url = url
    for _ in range(1000):  # hard stop
        r = requests.get(next_url, headers=_auth_headers(), timeout=20)
        r.raise_for_status()
        payload = r.json()
        chunk = payload.get("items", payload if isinstance(payload, list) else [])
        if isinstance(chunk, list):
            items.extend(chunk)
        next_path = payload.get("nextPagePath")
        if not next_path:
            break
        # T212 returns absolute path; join with base
        next_url = API_BASE.rstrip("/") + next_path
        # be polite
        time.sleep(0.2)
    return items

# ---------------------------
# Data shaping
# ---------------------------
def _load_overrides() -> dict:
    if OVERRIDES_PATH.exists():
        try:
            return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _infer_yf_symbol(t212_ticker: str, overrides: dict) -> tuple[str | None, str]:
    """
    Map a Trading212 ticker to a Yahoo Finance symbol.
    Returns (yf_symbol_or_none, listing_ccy 'GBP'|'USD'|None)
    """
    t = (t212_ticker or "").strip().upper()

    # 1) Explicit override wins
    if t in overrides:
        v = overrides[t]
        if isinstance(v, dict):
            return v.get("yf"), v.get("ccy", "GBP")
        if isinstance(v, str):
            return v, "GBP"

    # 2) US listings (e.g. AAPL_US_EQ → AAPL, USD)
    if "_US_" in t:
        core = t.split("_")[0]
        return core, "USD"

    # 3) LSE listings
    #    T212 often shows something like 'AHTL_EQ', 'HLMAL_EQ', 'GAWL_EQ'.
    #    Strip suffixes; if a trailing 'L' remains (AHTL → AHT), drop it,
    #    then append '.L' for Yahoo.
    core = t.replace("_GBX", "").replace("_GB", "").replace("_EQ", "")
    core = core.split("_")[0]

    # NEW: drop one trailing 'L' (e.g. 'AHTL' → 'AHT', 'HLMA' stays 'HLMA')
    if core.endswith("L") and len(core) >= 4:
        core = core[:-1]

    if core and core.isalpha() and 1 <= len(core) <= 5:
        return f"{core}.L", "GBP"

    # Fallback: unknown
    return None, None

def _build_position_timeseries(orders: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """
    orders columns expected: ['ticker','side','filledQuantity','filledAt'].
    side: BUY/SELL. filledQuantity positive numbers.
    Returns DataFrame index=dates, columns=tickers, values=shares held.
    """
    if orders.empty:
        return pd.DataFrame(index=pd.date_range(start, end, freq="D"))

    # Normalize
    orders = orders.copy()
    orders["filledAt"] = pd.to_datetime(orders["filledAt"], errors="coerce", utc=True).dt.date
    orders = orders.dropna(subset=["filledAt", "ticker", "filledQuantity"])
    orders["signed_qty"] = orders.apply(
        lambda r: float(r["filledQuantity"]) * (1.0 if str(r.get("side","BUY")).upper().startswith("B") else -1.0),
        axis=1
    )

    # Cumulate by day
    daily_flows = (
        orders.groupby(["filledAt","ticker"], as_index=False)["signed_qty"].sum()
               .rename(columns={"filledAt":"date"})
    )

    # Calendar index
    idx = pd.date_range(start, end, freq="D").date
    tickers = sorted(daily_flows["ticker"].unique().tolist())
    mat = pd.DataFrame(0.0, index=idx, columns=tickers)

    for _, row in daily_flows.iterrows():
        d  = row["date"]
        tk = row["ticker"]
        q  = float(row["signed_qty"])
        # Add from d onwards (position step)
        mat.loc[mat.index >= d, tk] += q

    # Remove columns that are zero the whole time
    mat = mat.loc[:, (mat != 0).any(axis=0)]
    return mat

def _download_fx_usd_gbp(start: date, end: date) -> pd.Series:
    url = f"https://api.frankfurter.app/{start}..{end}"
    r = requests.get(url, params={"from":"USD","to":"GBP"}, timeout=20)
    r.raise_for_status()
    data = r.json().get("rates", {})
    fx = pd.DataFrame.from_dict(data, orient="index").rename(columns={"GBP":"usd_gbp"})
    fx.index = pd.to_datetime(fx.index).date
    return fx["usd_gbp"]

def _download_prices(yf_map: dict[str, tuple[str,str]], start: date, end: date) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns GBP prices: index = calendar dates, columns = original T212 tickers.
    Enforces float64 at every step and never multiplies pandas objects that could
    carry strings.
    """
    missing: list[str] = []
    cal_idx = pd.date_range(start, end, freq="D").date
    out = pd.DataFrame(index=cal_idx)  # empty; we'll add verified float64 columns

    if yf is None:
        raise RuntimeError("yfinance is not installed. Add it to requirements and pip install.")

    gbp_syms = [t for t,(y,ccy) in yf_map.items() if y and ccy == "GBP"]
    usd_syms = [t for t,(y,ccy) in yf_map.items() if y and ccy == "USD"]

    def _dl(symbols: list[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        ysyms = [yf_map[t][0] for t in symbols]
        df = yf.download(
            ysyms,
            start=str(start),
            end=str(end + timedelta(days=1)),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df.index = pd.to_datetime(df.index).date
        # coerce every column to float64 (strings -> NaN)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        return df.reindex(cal_idx)
    # download by currency
    gbp_px = _dl(gbp_syms)
    usd_px = _dl(usd_syms)

    # ECB USD->GBP
    fx = _download_fx_usd_gbp(start, end) if not usd_px.empty else pd.Series(dtype="float64")
    fx = pd.to_numeric(fx, errors="coerce").astype("float64").reindex(cal_idx)
    fx_np = fx.to_numpy(dtype=np.float64, na_value=np.nan) if not fx.empty else None

    # ---- GBP listings (GBX->GBP heuristic) ----
    for t in gbp_syms:
        ysym = yf_map[t][0]
        ser = gbp_px.get(ysym)
        if ser is None or ser.dropna().empty:
            missing.append(t); continue
        med = float(ser.dropna().median()) if ser.notna().any() else np.nan
        if med and med > 1000:
            ser = ser / 100.0
        # final, guaranteed float64 column aligned to calendar
        out[t] = pd.to_numeric(ser, errors="coerce").astype("float64").reindex(cal_idx)

    # ---- USD listings (NumPy multiply only) ----
    if not usd_px.empty and fx_np is not None:
        for t in usd_syms:
            ysym = yf_map[t][0]
            ser = usd_px.get(ysym)
            if ser is None or ser.dropna().empty:
                missing.append(t); continue
            ser = pd.to_numeric(ser, errors="coerce").astype("float64").reindex(cal_idx)
            ser_np = ser.to_numpy(dtype=np.float64, na_value=np.nan)
            out[t] = pd.Series(ser_np * fx_np, index=cal_idx, dtype="float64")

    # final safety: all cols float64
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    return out, missing

# ---------------------------
# Public function
# ---------------------------
def backfill_nav_from_orders(start: str = "2025-01-01", end: str | None = None) -> Path:
    Path("data/_backfill_called.txt").write_text(
        f"called at {datetime.utcnow().isoformat()}Z\n", encoding="utf-8"
    )
    """
    Build daily NAV (GBP) into data/nav_daily.csv using orders history and yfinance.
    Also writes data/backfill_report.json with any missing symbols.
    """
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end,   "%Y-%m-%d").date() if end else datetime.utcnow().date()

    # 1) Pull ALL orders since the dawn of time, so positions are accurate at d0
    fetch_from = "1970-01-01"   # or "2000-01-01" if you prefer
    url = f"{API_BASE}/api/v0/equity/history/orders?from={fetch_from}&to={d1}"
    items = _paged_get(url)

    if not items:
        raise RuntimeError("No order history returned from Trading212.")

    # Expect keys like: ticker, side, filledQuantity, filledAt (or placedAt/updatedAt)
    o = pd.json_normalize(items)
    # Keep only filled orders
    if "status" in o.columns:
        o = o[o["status"].astype(str).str.upper().eq("FILLED")]

    # -----------------------------
    # Standardize & prepare orders
    # -----------------------------
    # Your payload has dateModified/dateCreated; accept several names.
    time_col = next(
        (c for c in [
            "filledAt", "updatedAt", "lastUpdated", "placedAt", "dateTime",
            "dateModified", "dateCreated"
        ] if c in o.columns),
        None
    )
    if time_col is None:
        raise RuntimeError(f"Could not find a fill timestamp column in orders. Columns: {list(o.columns)}")

    # We need: ticker, filledQuantity, timestamp → 'filledAt', and BUY/SELL side.
    need_cols = ["ticker", "filledQuantity", time_col]
    missing = [c for c in need_cols if c not in o.columns]
    if missing:
        raise RuntimeError(f"Orders payload missing columns: {missing}")

    w = o[need_cols].copy().rename(columns={time_col: "filledAt"})

    # Coerce timestamp (strings or epoch-ms) → date
    if pd.api.types.is_numeric_dtype(w["filledAt"]):
        dt = pd.to_datetime(w["filledAt"], unit="ms", errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(w["filledAt"].astype(str).str.replace("Z", "", regex=False),
                            errors="coerce", utc=True)
    w["filledAt"] = dt.dt.date

    # Determine BUY/SELL side robustly:
    # 1) If fillType starts with SELL/BUY (e.g. SELL_LIMIT, BUY_MARKET), use that.
    # 2) Else use the sign of filledValue (negative => SELL).
    # 3) Else use sign of orderedValue.
    # 4) Else default BUY.
    side_from_ft = pd.Series(index=o.index, dtype="object")
    if "fillType" in o.columns:
        ft = o["fillType"].astype(str).str.upper()
        side_from_ft = np.where(ft.str.startswith("SELL"), "SELL",
                         np.where(ft.str.startswith("BUY"),  "BUY",  np.nan))
        side_from_ft = pd.Series(side_from_ft, index=o.index)

    side_from_val = pd.Series(index=o.index, dtype="object")
    if "filledValue" in o.columns:
        fv = pd.to_numeric(o["filledValue"], errors="coerce")
        side_from_val = np.where(fv > 0, "SELL", np.where(fv < 0, "BUY", np.nan))
        side_from_val = pd.Series(side_from_val, index=o.index)

    side_from_ov = pd.Series(index=o.index, dtype="object")
    if "orderedValue" in o.columns:
        ov = pd.to_numeric(o["orderedValue"], errors="coerce")
        side_from_ov  = np.where(ov > 0, "SELL", np.where(ov < 0, "BUY", np.nan))
        side_from_ov = pd.Series(side_from_ov, index=o.index)

    # Priority: fillType -> filledValue sign -> orderedValue sign -> BUY
    side_final = side_from_ft.fillna(side_from_val).fillna(side_from_ov).fillna("BUY")
    w["side"] = side_final

    # After: w["side"] = side_final
    w.loc[w["side"].isna(), "side"] = "BUY"

    # Build the daily position matrix
    pos = _build_position_timeseries(w[["ticker", "side", "filledQuantity", "filledAt"]], d0, d1)

    # 2) Map tickers → Yahoo symbols
    overrides = _load_overrides()
    mapping = {}
    for t in pos.columns:
        ysym, ccy = _infer_yf_symbol(t, overrides)
        mapping[t] = (ysym, ccy)

    # 3) Prices (GBP)
    prices, miss = _download_prices(mapping, d0, d1)

    # Keep only tickers that have prices
    keep = [t for t in pos.columns if t in prices.columns]
    pos = pos[keep]
    prices = prices[keep]

    # Align to full calendar and forward-fill
    full_idx = pd.date_range(d0, d1, freq="D").date
    prices = prices.sort_index().reindex(full_idx).ffill()
    pos    = pos.sort_index().reindex(full_idx).ffill().fillna(0.0)

    # Force numeric (columns only; index stays as dates)
    for df_name, df in (("prices", prices), ("pos", pos)):
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # if anything is still string/object, capture it and stop with a readable message
        bad = []
        for c in df.columns:
            if df[c].dtype.kind not in ("f", "i"):  # not float/int
                bad.append(c)
        if bad:
            dbg = {
                "frame": df_name,
                "bad_columns": bad,
                "sample": df[bad].head(5).astype(str).to_dict(orient="list"),
            }
            (DATA_DIR / "debug").mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "debug" / "dtype_fail.json").write_text(json.dumps(dbg, indent=2), encoding="utf-8")
            raise RuntimeError(f"{df_name} contains non-numeric columns: {bad} (see data/debug/dtype_fail.json)")

    # Cast to contiguous float64 arrays (no pandas ops beyond this point)
    pos_np    = pos.to_numpy(dtype=np.float64, na_value=np.nan)
    prices_np = prices.to_numpy(dtype=np.float64, na_value=np.nan)

    # Elementwise multiply, then row-sum (ignores NaN)
    nav_vals = np.nansum(pos_np * prices_np, axis=1)
    nav_vals = pd.Series(nav_vals, index=pd.Index(full_idx, name="date"), dtype="float64")

    # Save CSV
    nav = pd.DataFrame({"date": pd.to_datetime(full_idx).strftime("%Y-%m-%d"),
                        "nav_gbp": nav_vals.values})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nav.to_csv(NAV_CSV, index=False)

    REPORT.write_text(json.dumps({"missing_symbols": miss, "mapped": mapping}, indent=2),
                      encoding="utf-8")
    return NAV_CSV


    # 5) Report
    REPORT.write_text(json.dumps({"missing_symbols": miss, "mapped": mapping}, indent=2), encoding="utf-8")

    return NAV_CSV