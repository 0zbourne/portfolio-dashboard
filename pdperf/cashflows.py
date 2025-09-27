from pathlib import Path
import pandas as pd
import json

def build_cash_flows(transactions_json_path: Path) -> pd.DataFrame:
    ...

    """
    Extract external cash flows (deposits +, withdrawals -) in GBP.
    If we can't detect any, returns an empty DataFrame (safe).
    Output columns: date, amount_gbp, kind
    """
    if not transactions_json_path.exists():
        return pd.DataFrame(columns=["date", "amount_gbp", "kind"])

    raw = json.loads(transactions_json_path.read_text(encoding="utf-8"))
    items = raw.get("items", raw)
    if not isinstance(items, list) or not items:
        return pd.DataFrame(columns=["date", "amount_gbp", "kind"])

    df = pd.json_normalize(items)

    action_col = next((c for c in df.columns if c.lower() in {"action", "type"}), None)
    amount_col = next((c for c in df.columns
                       if "totalamount" in c.replace("_", "").lower()
                       or c.lower() in {"total", "amount"}), None)
    date_col = next((c for c in df.columns
                     if any(k in c.lower() for k in ("time", "date", "created", "settled"))), None)

    if not all([action_col, amount_col, date_col]):
        return pd.DataFrame(columns=["date", "amount_gbp", "kind"])

    df[action_col] = df[action_col].astype(str).str.lower()
    flows = df[df[action_col].str.contains("deposit|withdraw", na=False)].copy()
    if flows.empty:
        return pd.DataFrame(columns=["date", "amount_gbp", "kind"])

    flows["amount_gbp"] = pd.to_numeric(flows[amount_col], errors="coerce")
    flows.loc[flows[action_col].str.contains("withdraw"), "amount_gbp"] *= -1

    d = pd.to_datetime(flows[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    flows["date"] = d.dt.date
    return flows[["date", "amount_gbp"]].assign(kind="external")
