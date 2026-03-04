# jobs/backfill.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
import os, json, time
import pandas as pd
import requests
import numpy as np

def _t212_headers():
    key = os.getenv("T212_API_KEY", "").strip()
    secret = os.getenv("T212_API_SECRET", "").strip()

    if key and secret:
        import base64
        token = base64.b64encode(f"{key}:{secret}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}", "Accept": "application/json"}

    if key and not key.lower().startswith("apikey "):
        key = f"Apikey {key}"
    return {"Authorization": key, "Accept": "application/json"}

try:
    pd.options.mode.dtype_backend = "numpy"
except Exception:
    pass
try:
    pd.options.mode.string_storage = "python"
except Exception:
    pass

try:
    import yfinance as yf
except Exception:
    yf = None

API_BASE = os.getenv("T212_API_BASE", "https://live.trading212.com")
API_KEY = os.getenv("T212_API_KEY")

DATA_DIR = Path("data")
NAV_CSV = DATA_DIR / "nav_daily.csv"
REPORT = DATA_DIR / "backfill_report.json"
OVERRIDES_PATH = DATA_DIR / "ticker_overrides.json"
CURRENCY_CACHE_PATH = DATA_DIR / "currency_cache.json"


def _auth_headers():
    if not API_KEY:
        raise RuntimeError("Missing T212_API_KEY in environment.")
    return {"Authorization": API_KEY, "Accept": "application/json"}


def _paged_get(url: str):
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


def _load_overrides() -> dict:
    """
    Load ticker overrides from file.
    Format: {"TICKER": {"yf": "SYMBOL.L", "ccy": "GBX"}}
    """
    hardcoded = {}
    if OVERRIDES_PATH.exists():
        try:
            file_overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
            hardcoded.update(file_overrides)
        except Exception:
            pass
    return hardcoded


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


def _get_yf_symbol_from_t212(t212_ticker: str, overrides: dict) -> str | None:
    """
    Convert T212 ticker to yfinance symbol.
    Returns yfinance symbol or None if cannot map.
    """
    t = (t212_ticker or "").strip().upper()
    overrides_upper = {k.upper(): v for k, v in overrides.items()}
    
    # Check overrides first
    if t in overrides_upper:
        v = overrides_upper[t]
        if isinstance(v, dict):
            return v.get("yf")
        if isinstance(v, str):
            return v

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


def get_yf_currency(ysym: str, overrides: dict) -> str:
    """
    Query yfinance for the actual currency of a ticker.
    
    Returns:
        "GBX" - yfinance returns pence (divide by 100)
        "GBP" - yfinance returns pounds (no conversion)
        "USD" - yfinance returns US dollars (convert via FX)
        "EUR" - etc.
    """
    if not ysym:
        return "USD"
    
    # Check file cache first
    cache = _load_currency_cache()
    if ysym in cache:
        return cache[ysym]
    
    if yf is None:
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


def _build_position_timeseries(orders: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if orders.empty:
        idx = pd.date_range(start, end, freq="D").date
        return pd.DataFrame(index=idx, dtype="float64")

    orders = orders.copy()
    orders["filledAt"] = pd.to_datetime(orders["filledAt"], errors="coerce", utc=True).dt.date
    orders = orders.dropna(subset=["filledAt", "ticker", "filledQuantity"])

    # Convert side to numeric
    side_str = orders.get("side", "BUY").astype(str).str.upper()
    sign = np.where(side_str.str.startswith("S"), -1.0, 1.0)
    qty = pd.to_numeric(orders["filledQuantity"], errors="coerce").astype("float64")
    orders["signed_qty"] = qty * sign

    # Create a full calendar
    idx = pd.date_range(start, end, freq="D").date
    tickers = sorted(orders["ticker"].unique().tolist())
    mat = pd.DataFrame(0.0, index=idx, columns=tickers, dtype="float64")

    # For each day, calculate position by summing all trades up to that day
    for day in idx:
        for ticker in tickers:
            # Sum all trades for this ticker up to (and including) this day
            trades = orders[(orders["ticker"] == ticker) & (orders["filledAt"] <= day)]
            if not trades.empty:
                mat.loc[day, ticker] = trades["signed_qty"].sum()

    mat = mat.loc[:, (mat != 0).any(axis=0)].astype("float64")
    return mat


def _download_fx_usd_gbp(start: date, end: date) -> pd.Series:
    """Fetch USD to GBP exchange rate from Frankfurter API."""
    url = f"https://api.frankfurter.app/{start}..{end}"
    
    for attempt in range(3):
        try:
            r = requests.get(url, params={"from": "USD", "to": "GBP"}, timeout=30)
            r.raise_for_status()
            data = r.json().get("rates", {})
            fx = pd.DataFrame.from_dict(data, orient="index").rename(columns={"GBP": "usd_gbp"})
            fx.index = pd.to_datetime(fx.index).date
            return fx["usd_gbp"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < 2:
                time.sleep(5)
                continue
            raise RuntimeError(f"FX API unavailable after 3 attempts: {e}") from e

def _download_prices(
    yf_map: dict[str, tuple[str, str]], 
    start: date, 
    end: date
) -> tuple[pd.DataFrame, list[str]]:
    """
    Download historical prices from yfinance and convert to GBP.
    
    Args:
        yf_map: Dict mapping T212 ticker -> (yfinance symbol, currency)
                currency is "GBX", "GBP", "USD", etc. (fetched from yfinance)
        start: Start date
        end: End date
    
    Returns:
        (prices_df, missing_symbols) where prices are in GBP
    """
    missing: list[str] = []
    cal_idx = pd.date_range(start, end, freq="D").date
    out = pd.DataFrame(index=cal_idx, dtype="float64")

    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    def _dl(symbols: list[str]) -> pd.DataFrame:
        """Download prices from yfinance with retries."""
        if not symbols:
            return pd.DataFrame()

        ysyms = [yf_map[t][0] for t in symbols]

        last_err: Exception | None = None
        for sleep_s in (0, 2, 5, 10, 20, 40):
            if sleep_s:
                time.sleep(sleep_s)

            try:
                df = yf.download(
                    ysyms,
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

                close = close.copy()
                close.index = pd.to_datetime(close.index).date

                for c in close.columns:
                    close.loc[:, c] = pd.to_numeric(close[c], errors="coerce").astype("float64")

                return close.reindex(cal_idx)

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Yahoo download failed: {last_err}") from last_err

    # Group tickers by currency
    gbx_tickers = [t for t, (y, ccy) in yf_map.items() if y and ccy == "GBX"]
    gbp_tickers = [t for t, (y, ccy) in yf_map.items() if y and ccy == "GBP"]
    usd_tickers = [t for t, (y, ccy) in yf_map.items() if y and ccy == "USD"]
    other_tickers = [t for t, (y, ccy) in yf_map.items() if y and ccy not in ("GBX", "GBP", "USD")]

    # Download all at once (yfinance handles batch)
    all_tickers = gbx_tickers + gbp_tickers + usd_tickers + other_tickers
    raw_prices = _dl(all_tickers)

    # Fetch FX rates for USD conversion
    fx = _download_fx_usd_gbp(start, end) if usd_tickers else pd.Series(dtype="float64")
    fx = pd.to_numeric(fx, errors="coerce").astype("float64").reindex(cal_idx)
    fx_np = fx.to_numpy(dtype=np.float64, na_value=np.nan) if not fx.empty else None

    # Process each ticker based on its currency
    for t212_ticker, (ysym, ccy) in yf_map.items():
        if not ysym:
            missing.append(t212_ticker)
            continue

        ser = raw_prices.get(ysym)
        if ser is None or ser.dropna().empty:
            missing.append(t212_ticker)
            continue

        ser = pd.to_numeric(ser, errors="coerce").astype("float64").reindex(cal_idx)

        # Convert to GBP based on ACTUAL currency from yfinance
        if ccy == "GBX":
            # Pence to pounds - yfinance returns pence for .L tickers
            ser = ser / 100.0
        elif ccy == "GBP":
            # Already in pounds, no conversion
            pass
        elif ccy == "USD":
            # Convert USD to GBP via FX rate
            if fx_np is not None:
                ser_np = ser.to_numpy(dtype=np.float64, na_value=np.nan)
                ser = pd.Series(ser_np * fx_np, index=cal_idx, dtype="float64")
        elif ccy == "EUR":
            # EUR would need EUR/GBP rate - approximate for now
            if fx_np is not None:
                ser_np = ser.to_numpy(dtype=np.float64, na_value=np.nan)
                # Rough EUR/GBP = 0.85 (should fetch proper rate)
                ser = pd.Series(ser_np * fx_np * 0.85, index=cal_idx, dtype="float64")
        else:
            # Unknown currency, assume GBP
            pass

        out.loc[:, t212_ticker] = ser

    for c in out.columns:
        out.loc[:, c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    return out, missing


def backfill_nav_from_orders(start: str = "2025-01-01", end: str | None = None) -> Path:
    """
    Rebuild daily NAV from Trading212 order history.
    
    Fetches all filled orders, builds position timeseries, downloads
    historical prices, and calculates daily portfolio value in GBP.
    
    Currency is fetched from yfinance for each ticker (cached locally).
    """
    import traceback

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
                if prices is not None:
                    f.write("\n[prices.dtypes]\n")
                    f.write(str(prices.dtypes) + "\n")
        except Exception as _e:
            print("[WARN] failed to write _backfill_trace.txt:", _e)

    try:
        (DATA_DIR / "_backfill_called.txt").write_text(
            f"called at {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8"
        )

        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date() if end else datetime.now(timezone.utc).date()

        # Fetch all orders from T212
        fetch_from = "1970-01-01"
        url = f"{API_BASE}/api/v0/equity/history/orders?from={fetch_from}&to={d1}"
        items = _paged_get(url)
        if not items:
            raise RuntimeError("No order history returned from Trading212.")

        o = pd.json_normalize(items)

        # Filter to filled orders only
        status_col = next((c for c in ["status", "order.status"] if c in o.columns), None)
        if status_col:
            o = o[o[status_col].astype(str).str.upper().eq("FILLED")]

        # Find required columns
        time_col = next((c for c in ["fill.filledAt", "filledAt", "order.filledAt", "order.createdAt"] if c in o.columns), None)
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

        w = pd.DataFrame({
            "ticker": o[ticker_col],
            "filledQuantity": o[qty_col],
            "filledAt": o[time_col],
            "side": o[side_col] if side_col else "BUY",
        })

        # Build position timeseries
        pos = _build_position_timeseries(w[["ticker", "side", "filledQuantity", "filledAt"]], d0, d1)

        # Load overrides
        overrides = _load_overrides()
        
        # Map T212 tickers to yfinance symbols AND fetch currencies
        mapping: dict[str, tuple[str, str]] = {}
        currency_report: dict[str, dict] = {}
        
        print(f"[BACKFILL] Fetching currencies for {len(pos.columns)} tickers...")
        
        for t in pos.columns:
            ysym = _get_yf_symbol_from_t212(t, overrides)
            
            if ysym:
                # Check if override specifies currency
                override_ccy = None
                t_upper = t.upper()
                if t_upper in {k.upper(): v for k, v in overrides.items()}:
                    v = {k.upper(): v for k, v in overrides.items()}[t_upper]
                    if isinstance(v, dict):
                        override_ccy = v.get("ccy")
                
                # Use override currency if provided, otherwise fetch from yfinance
                if override_ccy:
                    ccy = override_ccy
                    print(f"  {t} -> {ysym} -> {ccy} (override)")
                else:
                    ccy = get_yf_currency(ysym, overrides)
                    print(f"  {t} -> {ysym} -> {ccy}")
                
                mapping[t] = (ysym, ccy)
                currency_report[t] = {"yf_symbol": ysym, "currency": ccy, "source": "override" if override_ccy else "yfinance"}
            else:
                mapping[t] = (None, None)
                currency_report[t] = {"yf_symbol": None, "currency": None, "source": "failed"}
                print(f"  {t} -> FAILED (no yfinance symbol)")

        # Download prices
        print(f"[BACKFILL] Downloading prices from {d0} to {d1}...")
        prices, miss = _download_prices(mapping, d0, d1)

        # Align positions and prices
        keep = [t for t in pos.columns if t in prices.columns]
        if not keep:
            raise RuntimeError("No overlapping tickers between positions and prices.")
        pos = pos[keep]
        prices = prices[keep]

        # Forward fill missing prices
        full_idx = pd.date_range(d0, d1, freq="D").date
        prices = prices.sort_index().reindex(full_idx).ffill().astype("float64")
        pos = pos.sort_index().reindex(full_idx).ffill().fillna(0.0).astype("float64")

        # Calculate NAV
        pos_np = pos.to_numpy(dtype=np.float64, na_value=np.nan)
        prices_np = prices.to_numpy(dtype=np.float64, na_value=np.nan)
        nav_vals = np.nansum(pos_np * prices_np, axis=1)

        nav = pd.DataFrame({
            "date": pd.to_datetime(full_idx).strftime("%Y-%m-%d"),
            "nav_gbp": nav_vals.astype("float64")
        })
        NAV_CSV.parent.mkdir(parents=True, exist_ok=True)
        nav.to_csv(NAV_CSV, index=False)

        # Write detailed report
        REPORT.write_text(json.dumps({
            "missing_symbols": miss,
            "mapped": {t: [y, c] for t, (y, c) in mapping.items()},
            "currency_details": currency_report
        }, indent=2), encoding="utf-8")
        
        print(f"[BACKFILL] Complete. NAV written to {NAV_CSV}")
        return NAV_CSV

    except Exception:
        _dump_trace("FAILED", pos=locals().get("pos"), prices=locals().get("prices"))
        raise
