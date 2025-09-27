from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

DATA_DIR = Path("data")
NAV_CSV = DATA_DIR / "nav_daily.csv"

def _anchor_date_iso():
    """UTC date; if Saturday/Sunday roll back to Friday."""
    d = datetime.utcnow().date()
    if d.weekday() == 5:  # Sat
        d = d - timedelta(days=1)
    elif d.weekday() == 6:  # Sun
        d = d - timedelta(days=2)
    return d.isoformat()

def append_today_snapshot_if_missing(df, path: Path = NAV_CSV):
    """
    Append/update today's NAV (GBP) to data/nav_daily.csv.
    Uses df['total_value_gbp'] column already computed in app.py.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    date_key = _anchor_date_iso()
    nav = float(pd.to_numeric(df["total_value_gbp"], errors="coerce").sum())

    if path.exists():
        nav_df = pd.read_csv(path, parse_dates=["date"])
        nav_df["date"] = nav_df["date"].dt.date.astype(str)
    else:
        nav_df = pd.DataFrame(columns=["date", "nav_gbp"])

    if (nav_df["date"] == date_key).any():
        nav_df.loc[nav_df["date"] == date_key, "nav_gbp"] = nav
    else:
        nav_df = pd.concat(
            [nav_df, pd.DataFrame([{"date": date_key, "nav_gbp": nav}])],
            ignore_index=True
        )

    nav_df = nav_df.sort_values("date")
    nav_df.to_csv(path, index=False)
    return nav_df
