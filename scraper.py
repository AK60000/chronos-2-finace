import re
import requests
import pandas as pd
from datetime import datetime, timedelta


def _parse_sina_symbol(symbol: str) -> str:
    if symbol.startswith("sh") or symbol.startswith("sz"):
        return symbol
    if symbol.startswith("6"):
        return f"sh{symbol}"
    return f"sz{symbol}"


def fetch_stock_daily(
    symbol: str,
    days: int = 365,
    timeout: int = 30,
) -> pd.DataFrame:
    sina_symbol = _parse_sina_symbol(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 60)

    url = (
        f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={sina_symbol}"
        f"&scale=240&ma=no&datalen={min(days + 30, 1023)}"
    )

    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    raw = resp.json()
    if not raw:
        raise ValueError(f"No data returned for symbol {sina_symbol}")

    records = []
    for item in raw:
        records.append({
            "date": item.get("day", item.get("date", "")),
            "open": float(item.get("open", 0)),
            "high": float(item.get("high", 0)),
            "low": float(item.get("low", 0)),
            "close": float(item.get("close", 0)),
            "volume": float(item.get("volume", 0)),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.tail(days).reset_index(drop=True)

    return df


def fetch_stock_name(symbol: str, timeout: int = 10) -> str:
    sina_symbol = _parse_sina_symbol(symbol)
    url = f"https://hq.sinajs.cn/list={sina_symbol}"
    headers = {"Referer": "https://finance.sina.com.cn"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.encoding = "gbk"
        match = re.search(r'hq_str_\w+="(.+?)"', resp.text)
        if match:
            parts = match.group(1).split(",")
            return parts[0] if parts else symbol
    except Exception:
        pass
    return symbol
