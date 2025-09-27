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
    Formula (for day t): r_t = NAV_t / (NAV_{t-1} + flow_t) - 1
    where flow_t is cash added(+) or withdrawn(−) **during** day t before close.
    Returns a DataFrame with columns: date, r_port.
    """
    if nav is None or len(nav) == 0:
        return pd.DataFrame(columns=["date", "r_port"])

    # clean & sort
    nav = pd.to_numeric(nav, errors="coerce").dropna().sort_index()
    nav.name = "nav"

    # flows → series indexed by date (same freq), default 0
    if flows is not None and not flows.empty:
        f = flows.copy()
        # tolerate date as str/date/datetime
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = f.dropna(subset=["date"])
        f["amount_gbp"] = pd.to_numeric(f["amount_gbp"], errors="coerce")
        flow_s = f.groupby("date")["amount_gbp"].sum()
    else:
        flow_s = pd.Series(dtype="float64")

    # align on nav dates
    flow_s = flow_s.reindex(nav.index, fill_value=0.0)

    denom = nav.shift(1) + flow_s
    r = nav / denom - 1.0
    # first day has no denom — set to 0 for continuity
    r.iloc[0] = 0.0
    out = pd.DataFrame({"date": nav.index, "r_port": r.values})
    return out

def cumulative_return(obj, start: str | None = None, end: str | None = None) -> float:
    """
    Total return over a window.
    - If 'obj' is a DataFrame containing column 'r_port', use that column.
    - If 'obj' is a Series, use it directly.
    Applies date filtering if start/end provided.
    """
    if isinstance(obj, pd.DataFrame):
        if "date" in obj.columns:
            df = obj.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if start: df = df[df["date"] >= pd.to_datetime(start)]
            if end:   df = df[df["date"] <= pd.to_datetime(end)]
            s = pd.to_numeric(df["r_port"], errors="coerce").dropna()
        else:
            # fallback to first numeric column
            s = pd.to_numeric(obj.select_dtypes(include=[np.number]).iloc[:, 0], errors="coerce").dropna()
    else:
        s = pd.to_numeric(pd.Series(obj), errors="coerce").dropna()

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
