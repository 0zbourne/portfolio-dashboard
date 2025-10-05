# jobs/fundamentals.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import json
import math
import pandas as pd
import numpy as np

# reuse mapping + data dir from your backfill
from jobs.backfill import DATA_DIR, _infer_yf_symbol, _load_overrides

# Optional dependency
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

FUND_JSON = DATA_DIR / "fundamentals.json"
FUND_AUDIT = DATA_DIR / "fundamentals_audit.csv"

# ---------- helpers ----------
def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _pick_row(df: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    """Find a row in a yfinance statement by fuzzy name (case-insensitive)."""
    if df is None or df.empty:
        return None
    idx = [str(i).strip().lower() for i in df.index]
    for k in keys:
        k2 = k.strip().lower()
        try:
            j = idx.index(k2)
            return df.iloc[j]
        except ValueError:
            continue
    # try contains
    for k in keys:
        k2 = k.strip().lower()
        for j, name in enumerate(idx):
            if k2 in name:
                return df.iloc[j]
    return None

def _sum_last_n(series: pd.Series, n: int) -> float | None:
    if series is None or series.empty:
        return None
    vals = pd.to_numeric(series.dropna().tail(n), errors="coerce")
    vals = vals[pd.notna(vals)]
    if len(vals) == 0:
        return None
    return float(vals.sum())

def _get_yf_statements(yf_sym: str) -> dict:
    t = yf.Ticker(yf_sym)
    # Quarterly preferred (TTM), fall back to annual
    inc_q = t.quarterly_financials
    bs_q  = t.quarterly_balance_sheet
    cf_q  = t.quarterly_cashflow

    inc_a = t.financials
    bs_a  = t.balance_sheet
    cf_a  = t.cashflow

    return {
        "inc_q": inc_q, "bs_q": bs_q, "cf_q": cf_q,
        "inc_a": inc_a, "bs_a": bs_a, "cf_a": cf_a,
    }

def _ttm_or_fy(vals: pd.Series, annual: pd.Series) -> float | None:
    """Sum last 4 quarters; if not enough data use last annual latest value."""
    ttm = _sum_last_n(vals, 4) if vals is not None else None
    if ttm is not None:
        return ttm
    if annual is not None and not annual.dropna().empty:
        return _safe_float(annual.dropna().iloc[0])
    return None

def _compute_metrics_for_symbol(yf_sym: str) -> dict:
    """Return dict with roce, gm, om, cc, ic, and basis ('TTM' or 'FY') or {} if not available."""
    try:
        s = _get_yf_statements(yf_sym)
    except Exception:
        return {}

    # Income
    inc_q = s["inc_q"]; inc_a = s["inc_a"]
    rev_q = _pick_row(inc_q, ["total revenue", "revenue"])
    gp_q  = _pick_row(inc_q, ["gross profit"])
    ebit_q= _pick_row(inc_q, ["operating income", "ebit"])
    int_q = _pick_row(inc_q, ["interest expense", "interest expense non operating"])

    rev_a = _pick_row(inc_a, ["total revenue", "revenue"])
    gp_a  = _pick_row(inc_a, ["gross profit"])
    ebit_a= _pick_row(inc_a, ["operating income", "ebit"])
    int_a = _pick_row(inc_a, ["interest expense", "interest expense non operating"])

    # Balance sheet
    bs_q = s["bs_q"]; bs_a = s["bs_a"]
    ta_q = _pick_row(bs_q, ["total assets"])
    cl_q = _pick_row(bs_q, ["total current liabilities"])
    ta_a = _pick_row(bs_a, ["total assets"])
    cl_a = _pick_row(bs_a, ["total current liabilities"])

    # Cashflow
    cf_q = s["cf_q"]; cf_a = s["cf_a"]
    cfo_q = _pick_row(cf_q, ["total cash from operating activities", "operating cash flow"])
    cap_q = _pick_row(cf_q, ["capital expenditures", "capital expenditure"])
    cfo_a = _pick_row(cf_a, ["total cash from operating activities", "operating cash flow"])
    cap_a = _pick_row(cf_a, ["capital expenditures", "capital expenditure"])

    # TTM values (fallback to annual)
    rev  = _ttm_or_fy(rev_q,  rev_a)
    gp   = _ttm_or_fy(gp_q,   gp_a)
    ebit = _ttm_or_fy(ebit_q, ebit_a)
    intexp = _ttm_or_fy(int_q, int_a)  # typically negative, we'll abs()
    ta   = _ttm_or_fy(ta_q,   ta_a)
    cl   = _ttm_or_fy(cl_q,   cl_a)
    cfo  = _ttm_or_fy(cfo_q,  cfo_a)
    capex= _ttm_or_fy(cap_q,  cap_a)

    basis = "TTM" if any(x is not None for x in [
        _sum_last_n(rev_q,4), _sum_last_n(gp_q,4), _sum_last_n(ebit_q,4)
    ]) else "FY"

    # Derived
    invested_capital = None
    if ta is not None and cl is not None:
        invested_capital = ta - cl
        if invested_capital is not None and invested_capital <= 0:
            invested_capital = None

    fcf = None
    if cfo is not None and capex is not None:
        # yfinance capex is often negative; Fundsmith treats FCF = CFO - CapEx
        fcf = cfo - capex

    # Ratios
    def _safe_ratio(num, den):
        if num is None or den is None or den == 0:
            return None
        return float(num) / float(den)

    roce = _safe_ratio(ebit, invested_capital)
    gm   = _safe_ratio(gp, rev)
    om   = _safe_ratio(ebit, rev)
    cc   = _safe_ratio(fcf, ebit)
    ic   = _safe_ratio(ebit, abs(intexp) if intexp is not None else None)

    return {
        "basis": basis,
        "roce": roce,
        "gm": gm,
        "om": om,
        "cc": cc,
        "ic": ic,
        # for audit
        "_rev": rev, "_gp": gp, "_ebit": ebit, "_ta": ta, "_cl": cl,
        "_cfo": cfo, "_capex": capex, "_int": intexp, "_ic_inv_cap": invested_capital,
    }

def _map_to_yahoo(weights_t212: dict[str, float]) -> dict[str, float]:
    """Use overrides + _infer_yf_symbol to resolve Yahoo symbols for current holdings."""
    ovr = _load_overrides()
    out = {}
    for t212, w in weights_t212.items():
        yf_sym, _ccy = _infer_yf_symbol(t212, ovr)
        if yf_sym:
            out[yf_sym] = out.get(yf_sym, 0.0) + float(w)
    return out

def _reweighted_mean(vals: dict[str, float], weights: dict[str, float]) -> float | None:
    # drop NAs and renormalize
    filt = {k: v for k, v in vals.items() if v is not None and k in weights and weights[k] > 0}
    if not filt:
        return None
    z = sum(weights[k] for k in filt)
    if z <= 0:
        return None
    return sum(filt[k] * weights[k] / z for k in filt)

# ---------- public API ----------
def ensure_fundamentals(weights_t212: dict[str, float]) -> Path:
    """
    Build fundamentals.json if missing or older than 7 days.
    weights_t212: {T212_symbol: weight as decimal (0..1)} based on current market value.
    """
    if yf is None:
        raise RuntimeError("yfinance is required for fundamentals.")
    FUND_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Cache policy: rebuild if older than 7 days
    if FUND_JSON.exists():
        age_days = (datetime.utcnow() - datetime.utcfromtimestamp(FUND_JSON.stat().st_mtime)).days
        if age_days < 7:
            return FUND_JSON

    # Map weights to Yahoo
    w_yf = _map_to_yahoo(weights_t212)

    per_tkr = {}
    audit_rows = []

    for yf_sym, w in w_yf.items():
        try:
            m = _compute_metrics_for_symbol(yf_sym)
        except Exception:
            m = {}
        if m:
            per_tkr[yf_sym] = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                               for k, v in m.items()}
            audit_rows.append({
                "symbol": yf_sym, "weight": w, **m
            })

    # Portfolio-weighted aggregates (exclude missing per metric)
    vals = {k: {s: v.get(k) for s, v in per_tkr.items()} for k in ["roce","gm","om","cc","ic"]}
    agg = {k: _reweighted_mean(vals[k], w_yf) for k in vals}

    out = {
        "asof": datetime.utcnow().strftime("%Y-%m-%d"),
        "source": "yfinance",
        "refresh_policy": "weekly_if_stale",
        "per_ticker": per_tkr,
        "portfolio_weighted": agg,
    }

    FUND_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(FUND_AUDIT, index=False)
    return FUND_JSON

def load_fundamentals() -> dict:
    if FUND_JSON.exists():
        return json.loads(FUND_JSON.read_text(encoding="utf-8"))
    return {}
