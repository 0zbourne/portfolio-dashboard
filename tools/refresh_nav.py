import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# tools/refresh_nav.py
import os
from datetime import date
from jobs.backfill import backfill_nav_from_orders

# Start date for your NAV series. Adjust if you want a longer history.
START = os.getenv("NAV_ANCHOR", "2025-01-01")

if __name__ == "__main__":
    out_path = backfill_nav_from_orders(start=START)
    print(f"[refresh_nav] Wrote: {out_path}")
    
