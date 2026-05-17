import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from functools import lru_cache


# ==================================================
# 全域設定
# ==================================================

FINMIND_TOKEN = ""


def set_finmind_token(token: str = ""):
    """
    由 app.py 傳入 FinMind Token。
    """
    global FINMIND_TOKEN
    FINMIND_TOKEN = token or ""


def get_finmind_token():
    """
    取得 FinMind Token。
    優先順序：
    1. 程式內 set_finmind_token()
    2. 環境變數 FINMIND_TOKEN
    """
    if FINMIND_TOKEN:
        return FINMIND_TOKEN

    return os.getenv("FINMIND_TOKEN", "")


# ==================================================
# 基本工具
# ==================================================

def clean_stock_id(stock_code):
    """
    將股票代號清理成純代號。
    例如：
    2330.TW -> 2330
    6488.TWO -> 6488
    """

    stock_code = str(stock_code).strip()
    stock_code = stock_code.replace(".TW", "")
    stock_code = stock_code.replace(".TWO", "")
    stock_code = stock_code.replace(".EMG", "")

    return stock_code


def safe_float(value, default=np.nan):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_get(obj, key, default=np.nan):
    try:
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        return obj[key]

    except Exception:
        return default


def fmt_num(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "無資料"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "無資料"


def fmt_pct(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "無資料"
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "無資料"


def fmt_int(value):
    try:
        if value is None or pd.isna(value):
            return "無資料"
        return f"{float(value):,.0f}"
    except Exception:
        return "無資料"


def market_name_zh(market):
    if market == "listed":
        return "上市"
    if market == "otc":
        return "上櫃"
    if market == "emerging":
        return "興櫃"
    return "未知市場"


# ==================================================
# FinMind API
# ==================================================

def finmind_get(dataset, stock_code=None, start_date=None, end_date=None):
    """
    FinMind API 通用函數。
    """

    url = "https://api.finmindtrade.com/api/v4/data"

    params = {
        "dataset": dataset
    }

    if stock_code is not None:
        params["data_id"] = clean_stock_id(stock_code)

    if start_date is not None:
        params["start_date"] = start_date

    if end_date is not None:
        params["end_date"] = end_date

    token = get_finmind_token()

    if token:
        params["token"] = token

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if "data" not in data:
        raise ValueError(f"FinMind 回傳格式異常：{data}")

    df = pd.DataFrame(data["data"])

    if df.empty:
        raise ValueError(f"{dataset} 抓不到資料，股票代號：{stock_code}")

    return df


# ==================================================
# 股票清單與市場辨識
# ==================================================

@lru_cache(maxsize=1)
def fetch_taiwan_stock_universe():
    """
    取得台股股票清單。
    支援上市、上櫃、興櫃。
    """

    df = finmind_get(dataset="TaiwanStockInfo")

    for col in ["stock_id", "stock_name", "type", "industry_category", "date"]:
        if col not in df.columns:
            df[col] = ""

    df["stock_id"] = df["stock_id"].astype(str)
    df["stock_name"] = df["stock_name"].astype(str)
    df["type"] = df["type"].astype(str)
    df["industry_category"] = df["industry_category"].astype(str)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    df = df.drop_duplicates(subset=["stock_id"], keep="last")

    df["market"] = df.apply(normalize_market_type, axis=1)

    # 只保留常見 4 碼股票，排除 ETF 類型
    df = df[df["stock_id"].str.match(r"^\d{4}$", na=False)].copy()
    df = df[~df["stock_id"].str.startswith("00")].copy()

    exclude_keywords = [
        "ETF",
        "etf",
        "ETN",
        "etn",
        "基金",
        "受益證券",
        "指數股票"
    ]

    pattern = "|".join(exclude_keywords)

    df = df[
        ~df["stock_name"].str.contains(pattern, na=False)
        & ~df["industry_category"].str.contains(pattern, na=False)
    ].copy()

    df = df.sort_values(["market", "stock_id"]).reset_index(drop=True)

    return df


def normalize_market_type(row):
    """
    將 FinMind 的 type / industry_category 轉成市場分類。
    """

    text = (
        str(row.get("type", "")) + " "
        + str(row.get("industry_category", ""))
    ).lower()

    if "twse" in text or "上市" in text:
        return "listed"

    if "tpex" in text or "otc" in text or "上櫃" in text:
        return "otc"

    if "emerging" in text or "興櫃" in text:
        return "emerging"

    return "unknown"


def resolve_stock_input(stock_input):
    """
    支援輸入：
    1. 股票代號，例如 2330
    2. 股票代號加後綴，例如 2330.TW、6488.TWO
    3. 股票名稱，例如 台積電
    """

    raw = str(stock_input).strip()
    cleaned = clean_stock_id(raw)

    try:
        df = fetch_taiwan_stock_universe()
    except Exception:
        return {
            "stock_id": cleaned,
            "stock_name": "",
            "market": "unknown"
        }

    # 代號精準搜尋
    exact_id = df[df["stock_id"] == cleaned]

    if not exact_id.empty:
        row = exact_id.iloc[0]
        return {
            "stock_id": row["stock_id"],
            "stock_name": row["stock_name"],
            "market": row["market"]
        }

    # 名稱精準搜尋
    exact_name = df[df["stock_name"] == raw]

    if not exact_name.empty:
        row = exact_name.iloc[0]
        return {
            "stock_id": row["stock_id"],
            "stock_name": row["stock_name"],
            "market": row["market"]
        }

    # 名稱模糊搜尋
    name_match = df[df["stock_name"].str.contains(raw, na=False)]

    if len(name_match) == 1:
        row = name_match.iloc[0]
        return {
            "stock_id": row["stock_id"],
            "stock_name": row["stock_name"],
            "market": row["market"]
        }

    if len(name_match) > 1:
        candidates = name_match[["stock_id", "stock_name", "market", "industry_category"]].head(20)
        raise ValueError(
            "找到多筆相似股票，請改用股票代號。候選資料：\n"
            + candidates.to_string(index=False)
        )

    return {
        "stock_id": cleaned,
        "stock_name": "",
        "market": "unknown"
    }


def format_tw_stock_code(stock_input):
    """
    產生顯示用代號。
    """

    info = resolve_stock_input(stock_input)

    stock_id = info["stock_id"]
    market = info["market"]

    if market == "listed":
        return f"{stock_id}.TW"

    if market == "otc":
        return f"{stock_id}.TWO"

    if market == "emerging":
        return f"{stock_id}.EMG"

    return f"{stock_id}.TW"


# ==================================================
# 股價資料
# ==================================================

def fetch_stock_price_finmind(stock_input, years=3):
    """
    用 FinMind 抓股價資料。
    優先支援台股上市、上櫃、興櫃。
    """

    info = resolve_stock_input(stock_input)
    stock_id = info["stock_id"]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    df = finmind_get(
        dataset="TaiwanStockPrice",
        stock_code=stock_id,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    df["date"] = pd.to_datetime(df["date"])

    rename_map = {
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume"
    }

    for old_col in rename_map.keys():
        if old_col not in df.columns:
            df[old_col] = np.nan

    df = df.rename(columns=rename_map)

    use_cols = ["date", "Open", "High", "Low", "Close", "Volume"]
    df = df[use_cols].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.sort_values("date")
    df = df.set_index("date")

    if df.empty:
        raise ValueError(f"FinMind 股價資料為空：{stock_id}")

    symbol = format_tw_stock_code(stock_id)

    return symbol, df


def fetch_stock_price_yfinance_backup(stock_input, period="3y"):
    """
    yfinance 備援。
    主要支援上市 .TW、上櫃 .TWO。
    """

    info = resolve_stock_input(stock_input)

    stock_id = info["stock_id"]
    market = info["market"]

    if market == "listed":
        candidate_symbols = [f"{stock_id}.TW"]

    elif market == "otc":
        candidate_symbols = [f"{stock_id}.TWO"]

    elif market == "emerging":
        candidate_symbols = [f"{stock_id}.TWO", f"{stock_id}.TW"]

    else:
        candidate_symbols = [f"{stock_id}.TW", f"{stock_id}.TWO"]

    last_error = None

    for symbol in candidate_symbols:
        try:
            df = yf.download(
                symbol,
                period=period,
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()

            if not df.empty:
                return symbol, df

        except Exception as e:
            last_error = e

    raise ValueError(f"yfinance 抓不到資料：{candidate_symbols}，錯誤：{last_error}")


def fetch_stock_data(stock_input, period="3y"):
    """
    主股價函數。
    1. 先用 FinMind
    2. 失敗再用 yfinance
    """

    info = resolve_stock_input(stock_input)
    stock_id = info["stock_id"]

    try:
        return fetch_stock_price_finmind(stock_id, years=3)

    except Exception as e1:
        try:
            return fetch_stock_price_yfinance_backup(stock_id, period=period)

        except Exception as e2:
            raise ValueError(
                f"股價資料抓取失敗：{stock_id}\n"
                f"FinMind 錯誤：{e1}\n"
                f"yfinance 錯誤：{e2}"
            )
