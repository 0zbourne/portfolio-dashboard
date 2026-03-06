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

# Cache files for debugging
POSITIONS_CACHE = DATA_DIR / "positions_cache.parquet"
PRICES_CACHE = DATA_DIR / "prices_cache.parquet"
MAPPING_CACHE = DATA_DIR / "mapping_cache.json"


def _auth_headers():
    if not API_KEY:
        raise RuntimeError("Missing T212_API_KEY in environment.")
    return {"Authorization": API_KEY, "Accept": "application/json"}


def _paged_get(url: str):
    items = []
    next_url = url
    for _ in range(1000):
        r = requests.get(next_url, headers=_t212_headers(), timeout=20)
        
        # Handle rate limiting BEFORE raise_for_status
        if r.status_code == 429:
            print("[WARN] Rate limited by Trading212, waiting 60 seconds...")
            time.sleep(60)
            continue
        
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
    
def _fetch_cash_balance() -> dict:
    """Fetch current cash balance from Trading212."""
    try:
        url = f"{API_BASE}/api/v0/equity/account/cash"
        r = requests.get(url, headers=_t212_headers(), timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] Failed to fetch cash balance: {e}")
        return {}

def _fetch_transactions(start: date, end: date) -> list:
    """Fetch all transactions (orders, dividends, deposits, interest) from Trading212."""
    try:
        # Try the transactions endpoint
        url = f"{API_BASE}/api/v0/history/transactions"
        params = {"from": str(start), "to": str(end)}
        r = requests.get(url, headers=_t212_headers(), timeout=20, params=params)
        r.raise_for_status()
        payload = r.json()
        items = payload.get("items", payload if isinstance(payload, list) else [])
        return items
    except Exception as e:
        print(f"[WARN] Failed to fetch transactions: {e}")
        return []

def _build_cash_ledger(transactions: list, start: date, end: date) -> pd.Series:
    """
    Build a cash ledger from transactions.
    Returns Series with index=dates, values=cumulative cash balance.
    """
    if not transactions:
        print("[WARN] No transactions to build cash ledger")
        # Return zero cash for all dates
        idx = pd.date_range(start, end, freq="D").date
        return pd.Series(0.0, index=idx)
    
    # Parse transactions into DataFrame
    rows = []
    for tx in transactions:
        action = tx.get("action", tx.get("Action", "")).strip().lower()
        time_str = tx.get("time", tx.get("Time", tx.get("date", "")))
        total = tx.get("total", tx.get("Total", 0))
        
        # Parse amount - handle various field names
        if total:
            amount = float(total) if not isinstance(total, dict) else float(total.get("GBP", 0))
        else:
            # Try to get amount from other fields
            amount = float(tx.get("amount", tx.get("Result", 0)))
        
        # Parse date
        try:
            tx_date = pd.to_datetime(time_str).date()
        except:
            continue
        
        # Determine cash flow direction
        cash_flow = 0.0
        if "sell" in action:
            cash_flow = abs(amount)  # Selling adds cash
        elif "buy" in action:
            cash_flow = -abs(amount)  # Buying removes cash
        elif "dividend" in action:
            cash_flow = abs(amount)  # Dividends add cash
        elif "deposit" in action:
            cash_flow = abs(amount)  # Deposits add cash
        elif "withdraw" in action:
            cash_flow = -abs(amount)  # Withdrawals remove cash
        elif "interest" in action:
            cash_flow = abs(amount)  # Interest adds cash
        else:
            # Unknown action - check if it's positive or negative
            if amount > 0:
                cash_flow = amount
            else:
                cash_flow = amount
        
        rows.append({"date": tx_date, "cash_flow": cash_flow})
    
    if not rows:
        idx = pd.date_range(start, end, freq="D").date
        return pd.Series(0.0, index=idx)
    
    df = pd.DataFrame(rows)
    df = df.groupby("date")["cash_flow"].sum().sort_index()
    
    # Build cumulative cash balance
    idx = pd.date_range(start, end, freq="D").date
    cash = pd.Series(0.0, index=idx)
    
    # Cumulative sum of cash flows
    cumsum = df.cumsum()
    
    # Map to daily index (forward fill)
    for d in idx:
        flows_on_or_before = cumsum[cumsum.index <= d]
        if len(flows_on_or_before) > 0:
            cash.loc[d] = flows_on_or_before.iloc[-1]
    
    print(f"[INFO] Built cash ledger: {len(rows)} transactions, final cash: £{cash.iloc[-1]:.2f}")
    return cash

def _load_overrides() -> dict:
    """Load ticker overrides from JSON file. Keys normalized to uppercase for matching."""
    if OVERRIDES_PATH.exists():
        try:
            raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
            # Normalize keys to upper case for case-insensitive matching
            return {k.strip().upper(): v for k, v in raw.items()}
        except Exception:
            pass
    return {}


def _get_yf_symbol_from_t212(t212_ticker: str) -> str | None:
    """
    Convert T212 ticker to yfinance symbol (backwards compatibility for fundamentals.py).
    Returns just the yf symbol, no currency.
    """
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
    core = t.replace("_GBX", "").replace("_GB", "").replace("_EQ", "")
    core = core.split("_")[0]

    # Drop trailing 'L' (e.g. 'AHTL' → 'AHT', 'HLMA' stays 'HLMA')
    if core.endswith("L") and len(core) >= 4:
        core = core[:-1]

    if core and core.isalpha() and 1 <= len(core) <= 5:
        return f"{core}.L", "GBP"

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

    # PRINT DEBUG
    print("=== DEBUG: ORDERS BEFORE PROCESSING ===")
    for _, r in orders[orders["ticker"] == "LSEl_EQ"].iterrows():
        print(f"  {r['side']} {r['filledQuantity']} @ {r['filledAt']}")
    print(f"Total LSEl_EQ orders: {len(orders[orders['ticker'] == 'LSEl_EQ'])}")

    # BUY = +qty, SELL = -qty
    # Take absolute value because T212 returns negative quantities for SELLs
    side_str = orders.get("side", "BUY").astype(str).str.upper()
    sign = np.where(side_str.str.startswith("S"), -1.0, 1.0)
    qty = pd.to_numeric(orders["filledQuantity"], errors="coerce").astype("float64").abs()
    orders["signed_qty"] = qty * sign

    daily = (orders.groupby(["filledAt", "ticker"], as_index=False)["signed_qty"]
                    .sum().rename(columns={"filledAt": "date"}))

    # Debug: Check daily aggregation
    print("=== DEBUG: DAILY AGGREGATION CHECK ===")
    lseg_daily = daily[daily["ticker"] == "LSEl_EQ"]
    print(f"Columns: {daily.columns.tolist()}")
    print(f"LSEl_EQ daily data:")
    print(lseg_daily.to_string())
    print(f"signed_qty values: {lseg_daily['signed_qty'].tolist()}")
    
    # PRINT DEBUG
    print("=== DEBUG: DAILY CHANGES FOR LSEl_EQ ===")
    for _, r in daily[daily["ticker"] == "LSEl_EQ"].iterrows():
        print(f"  {r['date']}: {r['signed_qty']}")
    print(f"Total LSEl_EQ daily changes: {len(daily[daily['ticker'] == 'LSEl_EQ'])}")

    idx = pd.date_range(start, end, freq="D").date
    tickers = sorted(daily["ticker"].unique().tolist())

    # start as pure float64
    mat = pd.DataFrame(0.0, index=idx, columns=tickers, dtype="float64")

    # step function positions (holdings carry forward)
    for _, r in daily.iterrows():
        d = r["date"]; tk = r["ticker"]; q = float(r["signed_qty"])
        mat.loc[mat.index >= d, tk] += q

    # PRINT DEBUG
    print("=== DEBUG: FINAL POSITIONS FOR LSEl_EQ ===")
    lseg_pos = mat["LSEl_EQ"].dropna()
    for d, v in lseg_pos.items():
        if v != 0:
            print(f"  {d}: {v}")
    print(f"Final LSEl_EQ position: {lseg_pos.iloc[-1] if len(lseg_pos) > 0 else 0}")

    # drop all-zero columns and guarantee float64
    mat = mat.loc[:, (mat != 0).any(axis=0)].astype("float64")
    return mat


def _download_fx_usd_gbp(start: date, end: date) -> pd.Series:
    url = f"https://api.frankfurter.app/{start}..{end}"
    r = requests.get(url, params={"from": "USD", "to": "GBP"}, timeout=20)
    r.raise_for_status()
    data = r.json().get("rates", {})
    fx = pd.DataFrame.from_dict(data, orient="index").rename(columns={"GBP": "usd_gbp"})
    fx.index = pd.to_datetime(fx.index).date
    return fx["usd_gbp"]


def _download_prices(yf_map: dict[str, tuple[str, str]], start: date, end: date) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns GBP prices: index = calendar dates, columns = original T212 tickers.
    """
    missing: list[str] = []
    cal_idx = pd.date_range(start, end, freq="D").date
    out = pd.DataFrame(index=cal_idx, dtype="float64")

    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    gbp_syms = [t for t, (y, ccy) in yf_map.items() if y and ccy == "GBP"]
    usd_syms = [t for t, (y, ccy) in yf_map.items() if y and ccy == "USD"]

    def _dl(symbols: list[str]) -> pd.DataFrame:
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

                close.index = pd.to_datetime(close.index).date

                for c in close.columns:
                    close[c] = pd.to_numeric(close[c], errors="coerce").astype("float64")

                return close.reindex(cal_idx)

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Yahoo download failed: {last_err}") from last_err

    gbp_px = _dl(gbp_syms)
    usd_px = _dl(usd_syms)

    fx = _download_fx_usd_gbp(start, end) if usd_syms else pd.Series(dtype="float64")
    fx = pd.to_numeric(fx, errors="coerce").astype("float64").reindex(cal_idx).ffill()
    fx = fx.bfill()  # Fill any remaining NaN at the start
    fx_np = fx.to_numpy(dtype=np.float64, na_value=np.nan) if not fx.empty else None

    # ---- GBP listings (yfinance returns PENCE for .L stocks) ----
    for t in gbp_syms:
        ysym = yf_map[t][0]
        ser = gbp_px.get(ysym)
        if ser is None or ser.dropna().empty:
            missing.append(t)
            continue

        # yfinance returns PENCE for .L stocks - ALWAYS divide by 100
        if ysym.endswith(".L"):
            ser = ser / 100.0

        out[t] = pd.to_numeric(ser, errors="coerce").astype("float64").reindex(cal_idx).ffill().bfill()

    # ---- USD listings ----
    if not usd_px.empty and fx_np is not None:
        for t in usd_syms:
            ysym = yf_map[t][0]
            ser = usd_px.get(ysym)
            if ser is None or ser.dropna().empty:
                missing.append(t)
                continue
            ser = pd.to_numeric(ser, errors="coerce").astype("float64").reindex(cal_idx).ffill().bfill()
            ser_np = ser.to_numpy(dtype=np.float64, na_value=np.nan)
            out[t] = pd.Series(ser_np * fx_np, index=cal_idx, dtype="float64")

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    return out, missing


def backfill_nav_from_orders(start: str = "2025-01-01", end: str | None = None) -> Path:
    """
    Rebuild nav_daily.csv using Trading212 orders + yfinance prices.
    Also saves positions and prices cache for debugging.
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
                    bad = pos.select_dtypes(exclude=["float64"])
                    if not bad.empty:
                        f.write("non_float_positions: " + str(bad.columns.tolist()) + "\n")
                if prices is not None:
                    f.write("\n[prices.dtypes]\n")
                    f.write(str(prices.dtypes) + "\n")
                    bad = prices.select_dtypes(exclude=["float64"])
                    if not bad.empty:
                        f.write("non_float_prices: " + str(bad.columns.tolist()) + "\n")
        except Exception as _e:
            print("[WARN] failed to write _backfill_trace.txt:", _e)

    try:
        (DATA_DIR / "_backfill_called.txt").write_text(
            f"called at {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8"
        )

        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date() if end else datetime.now(timezone.utc).date() - timedelta(days=1)

        # 1) Orders from Trading212
        fetch_from = "1970-01-01"
        url = f"{API_BASE}/api/v0/equity/history/orders?from={fetch_from}&to={d1}"
        items = _paged_get(url)

        # Debug: Show raw orders for LSEl_EQ
        print("=== DEBUG: RAW ORDERS FOR LSEl_EQ ===")
        lseg_orders = [o for o in items if o.get("ticker") == "LSEl_EQ"]
        for i, o in enumerate(lseg_orders):
            print(f"{i+1}. {o.get('side')} {o.get('filledQuantity')} @ {o.get('filledAt')}")
        print(f"Total LSEl_EQ orders: {len(lseg_orders)}")
        
        if not items:
            raise RuntimeError("No order history returned from Trading212.")

        o = pd.json_normalize(items)

        # Filter to filled orders
        status_col = next((c for c in ["status", "order.status"] if c in o.columns), None)
        if status_col:
            o = o[o[status_col].astype(str).str.upper().eq("FILLED")]

        # Find columns
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

        # 2) Build positions timeseries
        pos = _build_position_timeseries(w[["ticker", "side", "filledQuantity", "filledAt"]], d0, d1)

        # 3) Map tickers to yfinance
        overrides = _load_overrides()
        mapping: dict[str, tuple[str, str]] = {}
        for t in pos.columns:
            ysym, ccy = _infer_yf_symbol(t, overrides)
            mapping[t] = (ysym, ccy)

        # 4) Download prices
        prices, miss = _download_prices(mapping, d0, d1)

        # 5) Align columns
        keep = [t for t in pos.columns if t in prices.columns]
        if not keep:
            raise RuntimeError("No overlapping tickers between positions and prices.")
        pos = pos[keep]
        prices = prices[keep]

        # === BUILD CASH TIMESERIES ===
        # === CASH TRACKING DISABLED ===
        # The old working version didn't track cash - positions * prices was enough
        # because trades are neutral (sell A, buy B = same NAV, just different holdings)
        cash_series = pd.Series(0.0, index=pos.index)

        # 6) Forward fill
        full_idx = pd.date_range(d0, d1, freq="D").date
        prices = prices.sort_index().reindex(full_idx).ffill().bfill().astype("float64")
        pos = pos.sort_index().reindex(full_idx).ffill().fillna(0.0).astype("float64")

        # === SAVE CACHES FOR DEBUGGING ===
        try:
            pos.to_parquet(POSITIONS_CACHE)
            prices.to_parquet(PRICES_CACHE)
            MAPPING_CACHE.write_text(json.dumps(mapping, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Failed to save debug caches: {e}")

        # 7) Calculate NAV (positions only, like the old working version)
        pos_np = pos.to_numpy(dtype=np.float64, na_value=np.nan)
        prices_np = prices.to_numpy(dtype=np.float64, na_value=np.nan)
        nav_vals = np.nansum(pos_np * prices_np, axis=1)

        nav = pd.DataFrame({
            "date": pd.to_datetime(full_idx).strftime("%Y-%m-%d"),
            "nav_gbp": nav_vals.astype("float64")
        })
        NAV_CSV.parent.mkdir(parents=True, exist_ok=True)
        nav.to_csv(NAV_CSV, index=False)

        REPORT.write_text(json.dumps({"missing_symbols": miss, "mapped": mapping}, indent=2), encoding="utf-8")
        return NAV_CSV

    except Exception:
        _dump_trace("FAILED", pos=locals().get("pos"), prices=locals().get("prices"))
        raise


def get_nav_breakdown(date_str: str) -> pd.DataFrame | None:
    """
    Get NAV breakdown for a specific date.
    Returns DataFrame with columns: ticker, yf_symbol, shares, price_gbp, value_gbp
    """
    try:
        if not POSITIONS_CACHE.exists() or not PRICES_CACHE.exists():
            return None

        pos = pd.read_parquet(POSITIONS_CACHE)
        prices = pd.read_parquet(PRICES_CACHE)
        mapping = json.loads(MAPPING_CACHE.read_text(encoding="utf-8")) if MAPPING_CACHE.exists() else {}

        # Parse date
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Find the date in the index
        if target_date not in pos.index:
            # Try to find closest date
            idx_dates = [d for d in pos.index if d <= target_date]
            if not idx_dates:
                return None
            target_date = max(idx_dates)

        # Get positions and prices for that date
        pos_row = pos.loc[target_date]
        price_row = prices.loc[target_date]

        # Build breakdown
        rows = []
        for ticker in pos.columns:
            shares = float(pos_row[ticker])
            if shares == 0:
                continue

            price = float(price_row[ticker]) if ticker in prices.columns else float("nan")
            value = shares * price if not np.isnan(price) else float("nan")

            yf_sym, ccy = mapping.get(ticker, (None, None))

            rows.append({
                "ticker": ticker,
                "yf_symbol": yf_sym or "N/A",
                "currency": ccy or "N/A",
                "shares": shares,
                "price_gbp": price,
                "value_gbp": value,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        # Sort by value descending
        df = df.sort_values("value_gbp", ascending=False, na_position="last").reset_index(drop=True)

        # Add total row
        total_row = pd.DataFrame([{
            "ticker": "TOTAL",
            "yf_symbol": "",
            "currency": "",
            "shares": "",
            "price_gbp": "",
            "value_gbp": df["value_gbp"].sum(),
        }])
        df = pd.concat([df, total_row], ignore_index=True)

        return df

    except Exception as e:
        print(f"[ERROR] get_nav_breakdown failed: {e}")
        return None
