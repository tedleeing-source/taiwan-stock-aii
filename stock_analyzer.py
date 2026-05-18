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


# ==================================================
# 技術指標與技術面評分
# ==================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    return macd, signal, hist


def add_indicators(df):
    df = df.copy()

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()
    df["MA120"] = df["Close"].rolling(window=120).mean()
    df["MA240"] = df["Close"].rolling(window=240).mean()

    df["RSI14"] = calculate_rsi(df["Close"], 14)

    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = calculate_macd(df["Close"])

    df["VOL20"] = df["Volume"].rolling(window=20).mean()
    df["VOL60"] = df["Volume"].rolling(window=60).mean()

    df["RET20"] = df["Close"].pct_change(20) * 100
    df["RET60"] = df["Close"].pct_change(60) * 100
    df["RET120"] = df["Close"].pct_change(120) * 100

    df["HIGH_120"] = df["High"].rolling(window=120).max()
    df["LOW_120"] = df["Low"].rolling(window=120).min()

    return df


def score_stock_technical(df):
    """
    技術面原始評分 100 分。
    最終完整模型會轉成 25 分。
    """

    clean_df = df.dropna()

    if len(clean_df) < 30:
        return {
            "score": 0,
            "details": ["技術面資料不足，無法分析"],
            "latest": None
        }

    latest = clean_df.iloc[-1]

    close = latest["Close"]
    ma60 = latest["MA60"]
    ma120 = latest["MA120"]
    ma240 = latest["MA240"]
    rsi = latest["RSI14"]
    macd = latest["MACD"]
    signal = latest["MACD_SIGNAL"]
    hist = latest["MACD_HIST"]
    vol = latest["Volume"]
    vol20 = latest["VOL20"]
    vol60 = latest["VOL60"]
    ret20 = latest["RET20"]
    high_120 = latest["HIGH_120"]

    score = 0
    details = []

    # 趨勢結構 45 分
    trend_score = 0

    if close > ma60:
        trend_score += 12
        details.append("股價站上 60 日均線，中期趨勢偏強")
    else:
        details.append("股價跌破 60 日均線，中期趨勢轉弱")

    if close > ma120:
        trend_score += 12
        details.append("股價站上 120 日均線，中長線趨勢偏多")
    else:
        details.append("股價跌破 120 日均線，中長線趨勢需保守看待")

    if close > ma240:
        trend_score += 8
        details.append("股價站上 240 日年線，長線結構偏多")
    else:
        details.append("股價仍在 240 日年線下方，長線結構尚未完全轉強")

    if ma60 > ma120:
        trend_score += 7
        details.append("60 日均線高於 120 日均線，均線結構偏多")
    else:
        details.append("60 日均線尚未高於 120 日均線，趨勢仍需觀察")

    if ma120 > ma240:
        trend_score += 6
        details.append("120 日均線高於 240 日均線，長線均線架構良好")
    else:
        details.append("120 日均線尚未高於 240 日均線，長線趨勢尚未完全確認")

    score += trend_score

    # 動能指標 20 分
    momentum_score = 0

    if macd > signal:
        momentum_score += 8
        details.append("MACD 位於 Signal 之上，動能偏多")
    else:
        details.append("MACD 低於 Signal，短中期動能偏弱")

    if hist > 0:
        momentum_score += 4
        details.append("MACD 柱狀體為正，買盤動能仍在")
    else:
        details.append("MACD 柱狀體為負，動能仍需觀察")

    if 45 <= rsi <= 70:
        momentum_score += 8
        details.append("RSI 位於健康偏強區間，尚未明顯過熱")
    elif 70 < rsi <= 80:
        momentum_score += 4
        details.append("RSI 偏高，短線有過熱風險")
    elif 30 <= rsi < 45:
        momentum_score += 4
        details.append("RSI 偏弱，但尚未進入嚴重超跌")
    elif rsi > 80:
        momentum_score += 2
        details.append("RSI 過熱，追價風險較高")
    else:
        momentum_score += 2
        details.append("RSI 偏低，股價動能明顯不足")

    score += momentum_score

    # 量能狀態 15 分
    volume_score = 0

    if vol20 > vol60:
        volume_score += 7
        details.append("20 日均量高於 60 日均量，量能有增溫跡象")
    else:
        details.append("20 日均量未高於 60 日均量，量能尚未明顯放大")

    if vol > vol20:
        volume_score += 5
        details.append("近期成交量高於 20 日均量，市場關注度提升")
    else:
        details.append("近期成交量低於 20 日均量，買盤力道較保守")

    if ret20 > 0 and vol20 > vol60:
        volume_score += 3
        details.append("近 20 日股價上漲且量能增溫，價量配合尚可")
    else:
        details.append("價量配合尚未明顯轉強")

    score += volume_score

    # 風險控制 20 分
    risk_score = 0

    distance_ma120 = ((close - ma120) / ma120) * 100
    drawdown_from_high = ((close - high_120) / high_120) * 100

    if distance_ma120 <= 30:
        risk_score += 8
        details.append("股價未明顯遠離 120 日均線，追高風險相對可控")
    else:
        details.append("股價大幅高於 120 日均線，短中期追高風險提高")

    if close > ma60:
        risk_score += 6
        details.append("股價仍守在 60 日均線上方，中期防線尚未跌破")
    else:
        details.append("股價跌破 60 日均線，中期風險升高")

    if drawdown_from_high > -20:
        risk_score += 6
        details.append("距離近 120 日高點回落幅度未超過 20%，趨勢尚未嚴重破壞")
    else:
        details.append("距離近 120 日高點回落超過 20%，需留意趨勢轉弱風險")

    score += risk_score

    score = min(round(score, 1), 100)

    return {
        "score": score,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "risk_score": risk_score,
        "details": details,
        "latest": latest,
        "distance_ma120": distance_ma120,
        "drawdown_from_high": drawdown_from_high
    }


# ==================================================
# 月營收成長動能
# ==================================================

def fetch_monthly_revenue(stock_input, years=4):
    """
    抓取月營收資料。
    """

    info = resolve_stock_input(stock_input)
    stock_id = info["stock_id"]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    df = finmind_get(
        dataset="TaiwanStockMonthRevenue",
        stock_code=stock_id,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    if "revenue_year" in df.columns and "revenue_month" in df.columns:
        df["period"] = pd.to_datetime(
            df["revenue_year"].astype(str)
            + "-"
            + df["revenue_month"].astype(str).str.zfill(2)
            + "-01"
        )
    else:
        df["period"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df = df.sort_values("period").drop_duplicates("period", keep="last")

    df["mom_pct"] = df["revenue"].pct_change(1) * 100
    df["yoy_pct"] = df["revenue"].pct_change(12) * 100
    df["avg_3m_yoy"] = df["yoy_pct"].rolling(3).mean()
    df["avg_6m_yoy"] = df["yoy_pct"].rolling(6).mean()

    return df


def score_monthly_revenue(df):
    """
    月營收成長評分 15 分。
    """

    clean_df = df.dropna(subset=["yoy_pct"])

    if clean_df.empty:
        return {
            "score": 0,
            "details": ["月營收資料不足，無法計算年增率"],
            "latest": None
        }

    latest = clean_df.iloc[-1]
    last_3 = clean_df.tail(3)

    latest_yoy = latest["yoy_pct"]
    avg_3m_yoy = latest["avg_3m_yoy"]
    avg_6m_yoy = latest["avg_6m_yoy"]

    score = 0
    details = []

    # 最新月營收年增率 6 分
    if latest_yoy >= 20:
        score += 6
        details.append("最新月營收年增率大於 20%，成長動能強")
    elif latest_yoy >= 10:
        score += 5
        details.append("最新月營收年增率大於 10%，成長動能偏多")
    elif latest_yoy >= 0:
        score += 3
        details.append("最新月營收年增率為正，營收仍維持成長")
    elif latest_yoy >= -5:
        score += 1
        details.append("最新月營收年增率小幅衰退，需觀察是否轉弱")
    else:
        details.append("最新月營收年增率明顯衰退，成長動能偏弱")

    # 近 3 個月平均年增率 4 分
    if pd.notna(avg_3m_yoy):
        if avg_3m_yoy >= 15:
            score += 4
            details.append("近 3 個月平均年增率大於 15%，營收趨勢良好")
        elif avg_3m_yoy >= 5:
            score += 3
            details.append("近 3 個月平均年增率為正，營收趨勢尚可")
        elif avg_3m_yoy >= 0:
            score += 1
            details.append("近 3 個月平均年增率小幅成長，動能普通")
        else:
            details.append("近 3 個月平均年增率為負，營收趨勢偏弱")
    else:
        details.append("近 3 個月平均年增率資料不足")

    # 近 3 個月皆正成長 2 分
    if len(last_3) >= 3 and (last_3["yoy_pct"] > 0).all():
        score += 2
        details.append("近 3 個月月營收年增率皆為正，成長穩定")
    else:
        details.append("近 3 個月月營收年增率並非全部為正，穩定性需觀察")

    # 近 3 個月優於近 6 個月 3 分
    if pd.notna(avg_3m_yoy) and pd.notna(avg_6m_yoy):
        if avg_3m_yoy > avg_6m_yoy:
            score += 3
            details.append("近 3 個月平均年增率高於近 6 個月，營收動能有升溫跡象")
        else:
            details.append("近 3 個月平均年增率未高於近 6 個月，營收動能尚未升溫")
    else:
        details.append("近 6 個月平均年增率資料不足")

    return {
        "score": min(score, 15),
        "details": details,
        "latest": latest
    }

# ==================================================
# 財報獲利品質分析
# EPS + ROE + 毛利率 + 營業利益率
# ==================================================

def fetch_financial_statements(stock_input, years=6):
    """
    抓取 FinMind 財報資料。
    用於分析：
    1. EPS
    2. ROE
    3. 毛利率
    4. 營業利益率
    """

    info = resolve_stock_input(stock_input)
    stock_id = info["stock_id"]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    df = finmind_get(
        dataset="TaiwanStockFinancialStatements",
        stock_code=stock_id,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "type" not in df.columns:
        df["type"] = ""

    if "origin_name" not in df.columns:
        df["origin_name"] = ""

    df["type_text"] = df["type"].astype(str)
    df["origin_text"] = df["origin_name"].astype(str)

    df["combined_text"] = (
        df["type_text"].str.lower()
        + " "
        + df["origin_text"].str.lower()
    )

    return df


def extract_financial_metric(
    df,
    exact_types=None,
    keywords=None,
    exclude_keywords=None
):
    """
    從財報資料中抓指定財務項目。

    使用方式：
    1. 先用 type 精準比對
    2. 再用 origin_name / type 關鍵字模糊搜尋
    3. 排除不需要的關鍵字
    """

    exact_types = exact_types or []
    keywords = keywords or []
    exclude_keywords = exclude_keywords or []

    if df.empty:
        return pd.Series(dtype=float)

    mask = pd.Series(False, index=df.index)

    # 1. type 精準比對
    if exact_types:
        exact_types_lower = [str(x).lower() for x in exact_types]

        mask_type = df["type_text"].str.lower().isin(exact_types_lower)
        mask = mask | mask_type

    # 2. 關鍵字搜尋
    if keywords:
        keyword_mask = pd.Series(False, index=df.index)

        for keyword in keywords:
            keyword_mask = keyword_mask | df["combined_text"].str.contains(
                str(keyword).lower(),
                na=False,
                regex=False
            )

        mask = mask | keyword_mask

    # 3. 排除關鍵字
    if exclude_keywords:
        for keyword in exclude_keywords:
            mask = mask & ~df["combined_text"].str.contains(
                str(keyword).lower(),
                na=False,
                regex=False
            )

    metric_df = df[mask].copy()

    if metric_df.empty:
        return pd.Series(dtype=float)

    metric_df = metric_df.dropna(subset=["value"])
    metric_df = metric_df.sort_values("date")

    metric_series = metric_df.groupby("date")["value"].last()
    metric_series = metric_series.sort_index()

    return metric_series


def build_profitability_table(stock_input):
    """
    建立獲利品質資料表。

    欄位包含：
    1. EPS
    2. 營收
    3. 毛利
    4. 營業利益
    5. 稅後淨利
    6. 股東權益
    7. 毛利率
    8. 營業利益率
    9. ROE
    """

    raw_df = fetch_financial_statements(stock_input)

    # EPS
    eps = extract_financial_metric(
        raw_df,
        exact_types=[
            "EPS",
            "BasicEPS",
            "EarningsPerShare"
        ],
        keywords=[
            "基本每股盈餘",
            "每股盈餘",
            "eps"
        ],
        exclude_keywords=[
            "稀釋"
        ]
    )

    # 營業收入
    revenue = extract_financial_metric(
        raw_df,
        exact_types=[
            "OperatingRevenue",
            "Revenue",
            "TotalRevenue"
        ],
        keywords=[
            "營業收入合計",
            "營業收入淨額",
            "營業收入",
            "operating revenue",
            "operatingrevenue",
            "revenue"
        ],
        exclude_keywords=[
            "營業外",
            "non-operating"
        ]
    )

    # 毛利
    gross_profit = extract_financial_metric(
        raw_df,
        exact_types=[
            "GrossProfit"
        ],
        keywords=[
            "營業毛利",
            "毛利",
            "gross profit",
            "grossprofit"
        ],
        exclude_keywords=[
            "毛利率"
        ]
    )

    # 營業利益
    operating_income = extract_financial_metric(
        raw_df,
        exact_types=[
            "OperatingIncome",
            "OperatingProfit"
        ],
        keywords=[
            "營業利益",
            "營業淨利",
            "operating income",
            "operatingincome",
            "operating profit"
        ],
        exclude_keywords=[
            "營業外",
            "non-operating",
            "營業利益率"
        ]
    )

    # 稅後淨利
    net_income = extract_financial_metric(
        raw_df,
        exact_types=[
            "NetIncome",
            "ProfitLoss",
            "NetIncomeAttributableToOwnersOfParent"
        ],
        keywords=[
            "本期淨利",
            "稅後淨利",
            "net income",
            "netincome",
            "profitloss"
        ],
        exclude_keywords=[
            "每股",
            "eps"
        ]
    )

    # 股東權益
    equity = extract_financial_metric(
        raw_df,
        exact_types=[
            "Equity",
            "TotalEquity",
            "EquityAttributableToOwnersOfParent"
        ],
        keywords=[
            "權益總額",
            "權益總計",
            "歸屬於母公司業主之權益",
            "equity attributable",
            "total equity",
            "equityattributabletoownersofparent",
            "equity"
        ],
        exclude_keywords=[
            "負債及權益",
            "liabilities and equity"
        ]
    )

    all_dates = sorted(
        set(eps.index)
        | set(revenue.index)
        | set(gross_profit.index)
        | set(operating_income.index)
        | set(net_income.index)
        | set(equity.index)
    )

    if not all_dates:
        raise ValueError("財報資料中找不到可用的獲利品質指標")

    fin_df = pd.DataFrame(index=all_dates)
    fin_df.index.name = "date"

    fin_df["eps"] = eps
    fin_df["revenue"] = revenue
    fin_df["gross_profit"] = gross_profit
    fin_df["operating_income"] = operating_income
    fin_df["net_income"] = net_income
    fin_df["equity"] = equity

    fin_df = fin_df.sort_index()

    # EPS 季增率
    fin_df["eps_qoq_pct"] = (
        (fin_df["eps"] - fin_df["eps"].shift(1))
        / fin_df["eps"].shift(1).abs()
    ) * 100

    # EPS 年增率
    fin_df["eps_yoy_pct"] = (
        (fin_df["eps"] - fin_df["eps"].shift(4))
        / fin_df["eps"].shift(4).abs()
    ) * 100

    # 近四季 EPS
    fin_df["ttm_eps"] = fin_df["eps"].rolling(4).sum()

    fin_df["ttm_eps_yoy_pct"] = (
        (fin_df["ttm_eps"] - fin_df["ttm_eps"].shift(4))
        / fin_df["ttm_eps"].shift(4).abs()
    ) * 100

    # 毛利率
    fin_df["gross_margin"] = (
        fin_df["gross_profit"] / fin_df["revenue"]
    ) * 100

    # 營業利益率
    fin_df["operating_margin"] = (
        fin_df["operating_income"] / fin_df["revenue"]
    ) * 100

    # ROE
    # 用近四季稅後淨利 / 平均股東權益估算
    fin_df["ttm_net_income"] = fin_df["net_income"].rolling(4).sum()
    fin_df["avg_equity"] = (fin_df["equity"] + fin_df["equity"].shift(4)) / 2
    fin_df["roe"] = (fin_df["ttm_net_income"] / fin_df["avg_equity"]) * 100

    # 近四季 EPS 是否為正
    fin_df["positive_eps_4q_count"] = (fin_df["eps"] > 0).rolling(4).sum()

    fin_df = fin_df.reset_index()

    return fin_df, raw_df


def score_profit_quality(fin_df):
    """
    獲利品質評分。
    滿分 25 分。

    評分項目：
    1. EPS：6 分
    2. ROE：7 分
    3. 毛利率：5 分
    4. 營業利益率：5 分
    5. 獲利穩定性：2 分
    """

    clean_df = fin_df.copy()

    useful_cols = [
        "eps",
        "roe",
        "gross_margin",
        "operating_margin"
    ]

    clean_df = clean_df.dropna(subset=useful_cols, how="all")

    if clean_df.empty:
        return {
            "score": 0,
            "details": ["財報資料不足，無法分析獲利品質"],
            "latest": None
        }

    latest = clean_df.iloc[-1]

    score = 0
    details = []

    eps = latest.get("eps", np.nan)
    eps_yoy = latest.get("eps_yoy_pct", np.nan)
    roe = latest.get("roe", np.nan)
    gross_margin = latest.get("gross_margin", np.nan)
    operating_margin = latest.get("operating_margin", np.nan)
    positive_eps_4q_count = latest.get("positive_eps_4q_count", np.nan)

    # EPS：6 分
    if pd.notna(eps):
        if eps > 0 and pd.notna(eps_yoy) and eps_yoy >= 20:
            score += 6
            details.append("最新 EPS 為正，且 EPS 年增率大於 20%，獲利成長強")
        elif eps > 0 and pd.notna(eps_yoy) and eps_yoy >= 0:
            score += 5
            details.append("最新 EPS 為正，且 EPS 年增率為正，獲利仍在成長")
        elif eps > 0:
            score += 4
            details.append("最新 EPS 為正，代表公司仍具獲利能力")
        elif eps == 0:
            score += 1
            details.append("最新 EPS 接近損益兩平，獲利能力普通")
        else:
            details.append("最新 EPS 為負，代表公司近期出現虧損")
    else:
        details.append("EPS 資料不足，無法判斷每股盈餘")

    # ROE：7 分
    if pd.notna(roe):
        if roe >= 15:
            score += 7
            details.append("ROE 高於 15%，資本報酬率優秀")
        elif roe >= 10:
            score += 5
            details.append("ROE 高於 10%，資本運用效率良好")
        elif roe >= 8:
            score += 3
            details.append("ROE 介於 8% 到 10%，資本報酬率尚可")
        elif roe > 0:
            score += 1
            details.append("ROE 為正但偏低，資本報酬率普通")
        else:
            details.append("ROE 為負，代表股東資本報酬表現偏弱")
    else:
        details.append("ROE 資料不足，可能是股東權益或淨利資料缺漏")

    # 毛利率：5 分
    if pd.notna(gross_margin):
        if gross_margin >= 40:
            score += 5
            details.append("毛利率高於 40%，產品競爭力強")
        elif gross_margin >= 25:
            score += 4
            details.append("毛利率高於 25%，產品具一定競爭力")
        elif gross_margin >= 15:
            score += 3
            details.append("毛利率介於 15% 到 25%，產品競爭力普通")
        elif gross_margin > 0:
            score += 1
            details.append("毛利率偏低，產品競爭力或成本控管需觀察")
        else:
            details.append("毛利率為負，代表本業成本壓力較大")
    else:
        details.append("毛利率資料不足，可能不適用於金融股或資料缺漏")

    # 營業利益率：5 分
    if pd.notna(operating_margin):
        if operating_margin >= 20:
            score += 5
            details.append("營業利益率高於 20%，本業獲利品質強")
        elif operating_margin >= 10:
            score += 4
            details.append("營業利益率高於 10%，本業獲利品質良好")
        elif operating_margin >= 5:
            score += 2
            details.append("營業利益率介於 5% 到 10%，本業獲利品質尚可")
        elif operating_margin > 0:
            score += 1
            details.append("營業利益率偏低，本業獲利能力普通")
        else:
            details.append("營業利益率為負，本業出現虧損")
    else:
        details.append("營業利益率資料不足，無法判斷本業獲利品質")

    # 獲利穩定性：2 分
    if pd.notna(positive_eps_4q_count):
        if positive_eps_4q_count == 4:
            score += 2
            details.append("近四季 EPS 皆為正，獲利穩定性佳")
        elif positive_eps_4q_count >= 3:
            score += 1
            details.append("近四季多數季度 EPS 為正，獲利穩定性尚可")
        else:
            details.append("近四季 EPS 穩定性不足，需觀察獲利波動")
    else:
        details.append("近四季 EPS 資料不足，無法判斷獲利穩定性")

    return {
        "score": min(score, 25),
        "details": details,
        "latest": latest
    }

# ==================================================
# 估值分析
# P/E + P/B + 殖利率
# ==================================================

def fetch_valuation_data(stock_input, years=5):
    """
    從 FinMind 抓取台股估值資料。

    包含：
    1. 本益比 PER / P/E
    2. 股價淨值比 PBR / P/B
    3. 殖利率 dividend_yield
    """

    info = resolve_stock_input(stock_input)
    stock_id = info["stock_id"]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    df = finmind_get(
        dataset="TaiwanStockPER",
        stock_code=stock_id,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    df["date"] = pd.to_datetime(df["date"])

    # 欄位名稱容錯處理
    if "PER" not in df.columns:
        possible_cols = [
            c for c in df.columns
            if c.lower() in ["per", "pe", "p_e"]
        ]

        if possible_cols:
            df["PER"] = df[possible_cols[0]]
        else:
            df["PER"] = np.nan

    if "PBR" not in df.columns:
        possible_cols = [
            c for c in df.columns
            if c.lower() in ["pbr", "pb", "p_b"]
        ]

        if possible_cols:
            df["PBR"] = df[possible_cols[0]]
        else:
            df["PBR"] = np.nan

    if "dividend_yield" not in df.columns:
        possible_cols = [
            c for c in df.columns
            if c.lower() in [
                "dividend_yield",
                "dividendyield",
                "yield",
                "殖利率"
            ]
        ]

        if possible_cols:
            df["dividend_yield"] = df[possible_cols[0]]
        else:
            df["dividend_yield"] = np.nan

    for col in ["PER", "PBR", "dividend_yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date")

    # 過濾無效值
    df["PER_valid"] = df["PER"].where(df["PER"] > 0)
    df["PBR_valid"] = df["PBR"].where(df["PBR"] > 0)

    # 歷史百分位
    # 百分位越低，代表相對歷史越便宜
    df["PER_percentile"] = df["PER_valid"].rank(pct=True) * 100
    df["PBR_percentile"] = df["PBR_valid"].rank(pct=True) * 100

    # 近一年、近三年平均
    df["PER_1y_avg"] = df["PER_valid"].rolling(240).mean()
    df["PBR_1y_avg"] = df["PBR_valid"].rolling(240).mean()

    df["PER_3y_avg"] = df["PER_valid"].rolling(720).mean()
    df["PBR_3y_avg"] = df["PBR_valid"].rolling(720).mean()

    return df


def score_valuation(valuation_df):
    """
    估值合理性評分。
    滿分 15 分。

    評分項目：
    1. 本益比 P/E：7 分
    2. 股價淨值比 P/B：5 分
    3. 殖利率：3 分
    """

    clean_df = valuation_df.copy()
    clean_df = clean_df.dropna(
        subset=["PER", "PBR", "dividend_yield"],
        how="all"
    )

    if clean_df.empty:
        return {
            "score": 0,
            "details": ["估值資料不足，無法分析 P/E、P/B 與殖利率"],
            "latest": None
        }

    latest = clean_df.iloc[-1]

    per = latest.get("PER", np.nan)
    pbr = latest.get("PBR", np.nan)
    dividend_yield = latest.get("dividend_yield", np.nan)

    per_percentile = latest.get("PER_percentile", np.nan)
    pbr_percentile = latest.get("PBR_percentile", np.nan)

    score = 0
    details = []

    # ==================================================
    # 1. 本益比 P/E：7 分
    # ==================================================

    pe_score = 0

    if pd.notna(per) and per > 0:
        # 歷史百分位判斷
        if pd.notna(per_percentile):
            if per_percentile <= 30:
                pe_score += 5
                details.append("目前本益比位於近年相對低檔，估值相對不貴")
            elif per_percentile <= 60:
                pe_score += 4
                details.append("目前本益比位於近年中間區間，估值大致合理")
            elif per_percentile <= 80:
                pe_score += 2
                details.append("目前本益比位於近年偏高區間，估值需留意")
            else:
                details.append("目前本益比位於近年高檔，追價風險較高")
        else:
            details.append("本益比歷史百分位資料不足，改用絕對本益比輔助判斷")

        # 絕對 P/E 輔助判斷
        if per <= 15:
            pe_score += 2
            details.append("本益比低於 15 倍，絕對估值偏低")
        elif per <= 25:
            pe_score += 2
            details.append("本益比低於 25 倍，絕對估值尚可")
        elif per <= 40:
            pe_score += 1
            details.append("本益比介於 25 到 40 倍，估值偏高但仍可觀察成長性")
        else:
            details.append("本益比高於 40 倍，需有高成長支撐，否則估值風險較高")
    else:
        details.append("本益比為負或無資料，可能代表近四季獲利不佳或資料缺漏")

    pe_score = min(pe_score, 7)
    score += pe_score

    # ==================================================
    # 2. 股價淨值比 P/B：5 分
    # ==================================================

    pb_score = 0

    if pd.notna(pbr) and pbr > 0:
        # 歷史百分位判斷
        if pd.notna(pbr_percentile):
            if pbr_percentile <= 30:
                pb_score += 3
                details.append("目前股價淨值比位於近年相對低檔，P/B 估值相對不貴")
            elif pbr_percentile <= 60:
                pb_score += 2
                details.append("目前股價淨值比位於近年中間區間，P/B 估值大致合理")
            elif pbr_percentile <= 80:
                pb_score += 1
                details.append("目前股價淨值比位於近年偏高區間，需留意評價壓力")
            else:
                details.append("目前股價淨值比位於近年高檔，評價風險較高")
        else:
            details.append("股價淨值比歷史百分位資料不足，改用絕對 P/B 輔助判斷")

        # 絕對 P/B 輔助判斷
        if pbr <= 1.5:
            pb_score += 2
            details.append("股價淨值比低於 1.5 倍，絕對 P/B 偏低")
        elif pbr <= 3:
            pb_score += 2
            details.append("股價淨值比低於 3 倍，絕對 P/B 尚可")
        elif pbr <= 5:
            pb_score += 1
            details.append("股價淨值比介於 3 到 5 倍，估值偏高但可觀察成長性")
        else:
            details.append("股價淨值比高於 5 倍，需留意股價相對淨值偏貴")
    else:
        details.append("股價淨值比為負或無資料，可能代表淨值異常或資料缺漏")

    pb_score = min(pb_score, 5)
    score += pb_score

    # ==================================================
    # 3. 殖利率：3 分
    # ==================================================

    if pd.notna(dividend_yield):
        if dividend_yield >= 5:
            score += 3
            details.append("殖利率高於 5%，具備較佳現金回饋吸引力")
        elif dividend_yield >= 3:
            score += 2
            details.append("殖利率高於 3%，具備一定現金回饋")
        elif dividend_yield > 0:
            score += 1
            details.append("公司有配息，但殖利率不高")
        else:
            details.append("殖利率為 0 或接近 0，現金股利吸引力較低")
    else:
        details.append("殖利率資料不足，無法判斷股利吸引力")

    score = min(score, 15)

    return {
        "score": score,
        "details": details,
        "latest": latest,
        "pe_score": pe_score,
        "pb_score": pb_score
    }

