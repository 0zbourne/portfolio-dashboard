# Ensure repo root on sys.path for CI
import sys, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Robust default for CI/local runs if env is missing/blank
START = (os.environ.get("NAV_ANCHOR") or "2025-01-01").strip()

# tools/refresh_nav.py
from datetime import date
from jobs.backfill import backfill_nav_from_orders

# Start date for your NAV series. Adjust if you want a longer history.
START = os.getenv("NAV_ANCHOR", "2025-01-01")

if __name__ == "__main__":
    out_path = backfill_nav_from_orders(start=START)
    print(f"[refresh_nav] Wrote: {out_path}")
    
