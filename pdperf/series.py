# perf/series.py

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
NAV_CSV = DATA_DIR / "nav_daily.csv"

def _read_csv_series(path: Path, value_col: str, date_col: str = "date") -> pd.Series:
    """
    Read a CSV with 'date' and one value column into a Series indexed by date (datetime64[ns]).
    Returns empty float series if file missing.
    """
    path = Path(path)
    if not path.exists():
        return pd.Series(dtype="float64")

    df = pd.read_csv(path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"{path} must contain columns '{date_col}' and '{value_col}'. Found: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    mask = df[date_col].notna() & vals.notna()
    df = df.loc[mask].sort_values(date_col)

    s = pd.Series(vals.loc[mask].values, index=df[date_col].values, name=value_col)
    s.index.name = date_col
    return s

def read_nav(path: Path = NAV_CSV) -> pd.Series:
    """Return daily portfolio NAV series (GBP) from data/nav_daily.csv (columns: date, nav_gbp)."""
    return _read_csv_series(path, value_col="nav_gbp", date_col="date")

def daily_returns_twr(nav: pd.Series, flows: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Time-weighted daily returns with external cash flows.
    Robust to leading zeros/NaNs in NAV so we don't produce infs.
    r_t = NAV_t / (NAV_{t-1} + flow_t) - 1
    """
    # Guard: empty
    if nav is None or len(nav) == 0:
        return pd.DataFrame(columns=["date", "r_port"])

    # Clean & sort
    nav = pd.to_numeric(nav, errors="coerce").sort_index()
    # Treat non-positive NAV as missing (cannot compute a return against 0)
    nav = nav.where(nav > 0)

    # Trim off leading NaNs/zeros so the series starts on the first valid NAV
    first_idx = nav.first_valid_index()
    if first_idx is None:
        return pd.DataFrame(columns=["date", "r_port"])
    nav = nav.loc[first_idx:]

    # ---- Flows (optional) ----
    if flows is not None and not flows.empty:
        f = flows.copy()
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = f.dropna(subset=["date"])
        f["amount_gbp"] = pd.to_numeric(f["amount_gbp"], errors="coerce")
        # Sum multiple events per calendar day
        per_day = f.groupby("date")["amount_gbp"].sum().sort_index()

        # Carry flows forward to the next valuation date:
        # 1) cumulative sum at flow dates
        cum = per_day.cumsum()

        # 2) map cumulative to NAV dates with forward-fill, then take differences between NAV dates
        cum_on_nav = cum.reindex(nav.index, method="ffill").fillna(0.0)
        flow_s = cum_on_nav.diff().fillna(cum_on_nav)
    else:
        # no flows at all
        flow_s = pd.Series(0.0, index=nav.index, dtype="float64")

    # Denominator: previous NAV + same-day flow
    denom = nav.shift(1) + flow_s

    # Compute daily return as a Series aligned to NAV index
    r = pd.Series(
        np.where(denom > 0, nav / denom - 1.0, 0.0),
        index=nav.index,
        dtype="float64",
    )

    # First day has no valid denominator: set to 0.0 for continuity
    if len(r) > 0:
        r.iloc[0] = 0.0

    # Drop non-finite values
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    # Emit DataFrame expected by the rest of the pipeline
    out = pd.DataFrame(
        {"date": r.index.astype("datetime64[ns]"), "r_port": r.values}
    )
    return out

def cumulative_return(obj, start: str | None = None, end: str | None = None) -> float:
    """
    Total return over a window.
    - If 'obj' is a DataFrame containing column 'r_port', use that column.
    - If 'obj' is a Series, use it directly.
    Applies date filtering if start/end provided.
    """
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if start:
                df = df[df["date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["date"] <= pd.to_datetime(end)]
            s = pd.to_numeric(df["r_port"], errors="coerce")
        else:
            s = pd.to_numeric(df.select_dtypes(include=[np.number]).iloc[:, 0], errors="coerce")
    else:
        s = pd.to_numeric(pd.Series(obj), errors="coerce")

    # Keep only finite values
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan

    return float((1.0 + s).prod() - 1.0)

def cagr(returns: pd.Series | pd.DataFrame, periods_per_year: int = 252) -> float:
    """
    Compound annual growth rate from a daily returns series or a DataFrame with 'r_port'.
    """
    if isinstance(returns, pd.DataFrame):
        s = pd.to_numeric(returns.get("r_port", returns.select_dtypes(include=[np.number]).iloc[:, 0]), errors="coerce").dropna()
    else:
        s = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()

    n = len(s)
    if n == 0:
        return np.nan
    total_growth = float((1.0 + s).prod())
    years = n / float(periods_per_year)
    if years <= 0:
        return np.nan
    return total_growth ** (1.0 / years) - 1.0
