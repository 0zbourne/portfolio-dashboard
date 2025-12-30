# jobs/backfill.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, date
import os, json, math, time
import pandas as pd
import requests
import numpy as np

def _t212_headers():
    """
    New Trading212 auth: Basic base64(KEY:SECRET)
    Legacy fallback kept for backwards compatibility.
    """
    key = os.getenv("T212_API_KEY", "").strip()
    secret = os.getenv("T212_API_SECRET", "").strip()

    # Preferred: Basic auth (key + secret)
    if key and secret:
        import base64
        token = base64.b64encode(f"{key}:{secret}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}", "Accept": "application/json"}

    # Legacy fallback: Apikey <key>
    if key and not key.lower().startswith("apikey "):
        key = f"Apikey {key}"
    return {"Authorization": key, "Accept": "application/json"}

# --- Hard-disable Arrow & nullable dtypes so everything is NumPy-native ---
try:
    pd.options.mode.dtype_backend = "numpy"         # pandas >=2.1
except Exception:
    pass
try:
    pd.options.mode.string_storage = "python"       # avoid pyarrow-backed strings
except Exception:
    pass

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
        r = requests.get(next_url, headers=_t212_headers(), timeout=20)
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
        time.sleep(2.1)
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
    Returns DataFrame index=dates, columns=tickers, values=shares held (float64).
    """
    if orders.empty:
        idx = pd.date_range(start, end, freq="D").date
        return pd.DataFrame(index=idx, dtype="float64")

    orders = orders.copy()
    orders["filledAt"] = pd.to_datetime(orders["filledAt"], errors="coerce", utc=True).dt.date
    orders = orders.dropna(subset=["filledAt", "ticker", "filledQuantity"])

    # BUY = +qty, SELL = -qty
    side_str = orders.get("side", "BUY").astype(str).str.upper()
    sign = np.where(side_str.str.startswith("S"), -1.0, 1.0)
    qty = pd.to_numeric(orders["filledQuantity"], errors="coerce").astype("float64")
    orders["signed_qty"] = qty * sign

    daily = (orders.groupby(["filledAt", "ticker"], as_index=False)["signed_qty"]
                    .sum().rename(columns={"filledAt": "date"}))

    idx = pd.date_range(start, end, freq="D").date
    tickers = sorted(daily["ticker"].unique().tolist())

    # start as pure float64
    mat = pd.DataFrame(0.0, index=idx, columns=tickers, dtype="float64")

    # step function positions (holdings carry forward)
    for _, r in daily.iterrows():
        d = r["date"]; tk = r["ticker"]; q = float(r["signed_qty"])
        mat.loc[mat.index >= d, tk] += q

    # drop all-zero columns and guarantee float64
    mat = mat.loc[:, (mat != 0).any(axis=0)].astype("float64")
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
    out = pd.DataFrame(index=cal_idx, dtype="float64")

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
        # --- robust unit check using yfinance metadata ---
        try:
            info = yf.Ticker(ysym).info
            if info.get("currency") == "GBp":
                # yfinance reports in pence → convert to pounds
                ser = ser / 100.0
        except Exception:
            # fallback heuristic if metadata is unavailable
            med = float(ser.dropna().median()) if ser.notna().any() else None
            if med and med > 1000.0:
                ser = ser / 100.0

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

def backfill_nav_from_orders(start: str = "2025-01-01", end: str | None = None) -> Path:
    """
    Rebuild nav_daily.csv using Trading212 orders + yfinance prices.
    On any failure, writes data/_backfill_trace.txt with the full traceback and any dtype info we reached.
    """
    # ---------- black-box crash logger ----------
    import json, traceback
    from datetime import datetime, timedelta

    trace_path = DATA_DIR / "_backfill_trace.txt"

    def _dump_trace(stage: str, pos=None, prices=None):
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write(f"[STAGE] {stage}\n\n")
                f.write("[TRACEBACK]\n")
                f.write(traceback.format_exc())
                f.write("\n")
                if pos is not None:
                    f.write("\n[positions.dtypes]\n")
                    f.write(str(pos.dtypes) + "\n")
                    bad = pos.select_dtypes(exclude=["float64"])
                    if not bad.empty:
                        f.write("non_float_positions: " + str(bad.columns.tolist()) + "\n")
                if prices is not None:
                    f.write("\n[prices.dtypes]\n")
                    f.write(str(prices.dtypes) + "\n")
                    bad = prices.select_dtypes(exclude=["float64"])
                    if not bad.empty:
                        f.write("non_float_prices: " + str(bad.columns.tolist()) + "\n")
        except Exception as _e:  # don't let logging fail the run
            print("[WARN] failed to write _backfill_trace.txt:", _e)

    # ---------- original logic (kept, but wrapped) ----------
    try:
        # marker so we know the button actually called us
        (DATA_DIR / "_backfill_called.txt").write_text(
            f"called at {datetime.utcnow().isoformat()}Z\n", encoding="utf-8"
        )

        # dates
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end,   "%Y-%m-%d").date() if end else datetime.utcnow().date()

        # 1) Orders from Trading212
        fetch_from = "1970-01-01"
        url = f"{API_BASE}/api/v0/equity/history/orders?from={fetch_from}&to={d1}"
        items = _paged_get(url)
        if not items:
            raise RuntimeError("No order history returned from Trading212.")

        o = pd.json_normalize(items)
        if "status" in o.columns:
            o = o[o["status"].astype(str).str.upper().eq("FILLED")]

        # --- NEW: handle Trading212's nested schema (order.* / fill.*) ---
        # Status column can be "status" (old) or "order.status" (new)
        status_col = next((c for c in ["status", "order.status"] if c in o.columns), None)
        if status_col:
            o = o[o[status_col].astype(str).str.upper().eq("FILLED")]

        # Prefer fill-level timestamp if present (matches fill qty/price)
        time_col = next((c for c in ["fill.filledAt", "filledAt", "order.filledAt", "order.createdAt"] if c in o.columns), None)
        # Fallback: anything ending in ".filledAt" or ending in "filledAt"
        if time_col is None:
            time_col = next((c for c in o.columns if str(c).endswith(".filledAt") or str(c).endswith("filledAt")), None)
        if time_col is None:
            raise RuntimeError(f"Could not find a fill timestamp column. Columns: {list(o.columns)}")

        ticker_col = next((c for c in ["order.ticker", "order.instrument.ticker", "ticker"] if c in o.columns), None)
        if ticker_col is None:
            raise RuntimeError(f"Could not find a ticker column. Columns: {list(o.columns)}")

        qty_col = next((c for c in ["fill.quantity", "order.filledQuantity", "filledQuantity"] if c in o.columns), None)
        if qty_col is None:
            raise RuntimeError(f"Could not find a filled quantity column. Columns: {list(o.columns)}")

        side_col = next((c for c in ["order.side", "side"] if c in o.columns), None)

        # Build canonical dataframe expected by downstream functions
        w = pd.DataFrame({
            "ticker": o[ticker_col],
            "filledQuantity": o[qty_col],
            "filledAt": o[time_col],
            "side": o[side_col] if side_col else "BUY",
        })

        # 2) Build positions timeseries (float64 matrix)
        pos = _build_position_timeseries(w[["ticker","side","filledQuantity","filledAt"]], d0, d1)

        # 3) Map tickers to yfinance & currencies (using your overrides)
        overrides = _load_overrides()
        mapping: dict[str, tuple[str,str]] = {}
        for t in pos.columns:
            ysym, ccy = _infer_yf_symbol(t, overrides)
            mapping[t] = (ysym, ccy)

        # 4) Download prices (GBP/GBp normalized, USD->GBP converted)
        prices, miss = _download_prices(mapping, d0, d1)

        # align columns present on both sides
        keep = [t for t in pos.columns if t in prices.columns]
        if not keep:
            raise RuntimeError("No overlapping tickers between positions and prices.")
        pos = pos[keep]
        prices = prices[keep]

        # 5) Align to full calendar and ffill; hard-cast float64
        full_idx = pd.date_range(d0, d1, freq="D").date
        prices = (prices.sort_index().reindex(full_idx).ffill().astype("float64"))
        pos    = (pos.sort_index().reindex(full_idx).ffill().fillna(0.0).astype("float64"))

        # quick dtype snapshot (if we reached here, file will exist)
        with open(DATA_DIR / "_dtype_snapshot.txt", "w", encoding="utf-8") as f:
            f.write("[positions.dtypes]\n")
            f.write(str(pos.dtypes) + "\n\n")
            f.write("[prices.dtypes]\n")
            f.write(str(prices.dtypes) + "\n\n")

        # 6) Multiply strictly on float64 arrays
        pos_np    = pos.to_numpy(dtype=np.float64, na_value=np.nan)
        prices_np = prices.to_numpy(dtype=np.float64, na_value=np.nan)
        nav_vals  = np.nansum(pos_np * prices_np, axis=1)

        nav = pd.DataFrame({
            "date": pd.to_datetime(full_idx).strftime("%Y-%m-%d"),
            "nav_gbp": nav_vals.astype("float64")
        })
        NAV_CSV.parent.mkdir(parents=True, exist_ok=True)
        nav.to_csv(NAV_CSV, index=False)

        REPORT.write_text(json.dumps({"missing_symbols": miss, "mapped": mapping}, indent=2),
                          encoding="utf-8")
        return NAV_CSV

    except Exception:
        _dump_trace("FAILED", pos=locals().get("pos"), prices=locals().get("prices"))
        raise
