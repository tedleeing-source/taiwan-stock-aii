import streamlit as st
import pandas as pd
import traceback

import stock_analyzer as sa


# ==================================================
# Streamlit 頁面設定
# ==================================================

st.set_page_config(
    page_title="台股中長線 AI 股票分析師",
    page_icon="📈",
    layout="wide"
)


# ==================================================
# 讀取 FinMind Token
# ==================================================

def get_secret_token():
    """
    從 Streamlit Secrets 讀取 FinMind Token。
    如果還沒設定，也可以空白執行，但可能會遇到 API 限制。
    """
    try:
        return st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        return ""


FINMIND_TOKEN = get_secret_token()
sa.set_finmind_token(FINMIND_TOKEN)


# ==================================================
# 小工具
# ==================================================

def show_score_card(title, score, max_score):
    """
    顯示分數卡片。
    """
    st.metric(
        label=title,
        value=f"{score} / {max_score}"
    )


def show_details(details):
    """
    顯示條件檢查。
    """
    if not details:
        st.info("無條件檢查資料")
        return

    for item in details:
        st.write(f"- {item}")


def safe_dataframe(df):
    """
    安全顯示 DataFrame。
    """
    if df is None:
        st.info("無資料")
        return

    if not isinstance(df, pd.DataFrame):
        st.info("資料格式不正確")
        return

    if df.empty:
        st.info("資料表為空")
        return

    st.dataframe(df, use_container_width=True)


# ==================================================
# 快取分析
# ==================================================

@st.cache_data(ttl=600, show_spinner=False)
def cached_analyze(stock_code, token):
    """
    快取 10 分鐘，避免短時間重複打 API。
    """
    sa.set_finmind_token(token)
    return sa.analyze_stock(stock_code)


# ==================================================
# 網站標題
# ==================================================

st.title("📈 台股中長線 AI 股票分析師")

st.markdown(
    """
    這是一個針對 **台股上市、上櫃、興櫃股票** 的中長線分析工具。  
    分析週期以 **3 到 6 個月** 為主，整合技術面、月營收、獲利品質、估值與三大法人籌碼。
    """
)

st.warning("本工具僅供研究參考，不構成任何投資建議。")


# ==================================================
# 側邊欄
# ==================================================

with st.sidebar:
    st.header("設定")

    if FINMIND_TOKEN:
        st.success("FinMind Token 已設定")
    else:
        st.warning("尚未設定 FinMind Token，可能會遇到 API 使用限制")

    st.markdown("---")

    st.markdown(
        """
        **支援輸入**
        - 股票代號：例如 `2330`
        - 上櫃股票：例如 `6488`
        - 股票名稱：例如 `台積電`
        """
    )

    st.markdown("---")

    st.markdown(
        """
        **評分模型**
        - 技術面：25 分
        - 月營收：15 分
        - 獲利品質：25 分
        - 估值合理性：15 分
        - 三大法人籌碼：20 分
        """
    )


# ==================================================
# 輸入區
# ==================================================

st.subheader("輸入股票代號")

with st.form("stock_form"):
    stock_code = st.text_input(
        "股票代號或股票名稱",
        value="2330",
        placeholder="例如：2330、6488、8299、台積電"
    )

    submitted = st.form_submit_button("開始分析")


# ==================================================
# 分析區
# ==================================================

if submitted:
    stock_code = stock_code.strip()

    if not stock_code:
        st.error("請輸入股票代號或股票名稱")
        st.stop()

    try:
        with st.spinner(f"正在分析 {stock_code}，請稍候..."):
            result = cached_analyze(stock_code, FINMIND_TOKEN)

        st.success("分析完成")

        stock_id = result["stock_id"]
        stock_name = result["stock_name"]
        market = result["market"]
        symbol = result["symbol"]
        total_score = result["total_score"]
        viewpoint = result["viewpoint"]
        conclusion = result["conclusion"]
        scores = result["scores"]
        metrics = result["metrics"]
        details = result["details"]
        errors = result["errors"]
        data = result["data"]
        report = result["report"]

        # ==================================================
        # 股票基本資訊
        # ==================================================

        st.markdown("---")
        st.subheader("分析結果總覽")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("股票代號", symbol)

        with col2:
            st.metric("股票名稱", stock_name if stock_name else "無資料")

        with col3:
            st.metric("市場", sa.market_name_zh(market))

        with col4:
            st.metric("中長線觀點", viewpoint)

        col_score_1, col_score_2 = st.columns([1, 2])

        with col_score_1:
            st.metric("整合總分", f"{total_score} / 100")

        with col_score_2:
            st.info(conclusion)

        # ==================================================
        # 分項分數
        # ==================================================

        st.markdown("---")
        st.subheader("分項評分")

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            show_score_card("技術面", scores["technical"], 25)

        with c2:
            show_score_card("月營收", scores["revenue"], 15)

        with c3:
            show_score_card("獲利品質", scores["profit_quality"], 25)

        with c4:
            show_score_card("估值", scores["valuation"], 15)

        with c5:
            show_score_card("三大法人", scores["institutional"], 20)

        # ==================================================
        # Tabs
        # ==================================================

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "技術面",
                "月營收",
                "獲利品質",
                "估值",
                "三大法人",
                "完整報告"
            ]
        )

        # ==================================================
        # Tab 1：技術面
        # ==================================================

        with tab1:
            st.subheader("技術面分析")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("目前收盤價", sa.fmt_num(metrics["close"]))

            with c2:
                st.metric("60 日均線", sa.fmt_num(metrics["ma60"]))

            with c3:
                st.metric("120 日均線", sa.fmt_num(metrics["ma120"]))

            with c4:
                st.metric("240 日均線", sa.fmt_num(metrics["ma240"]))

            c5, c6, c7, c8 = st.columns(4)

            with c5:
                st.metric("RSI 14", sa.fmt_num(metrics["rsi"]))

            with c6:
                st.metric("MACD", sa.fmt_num(metrics["macd"]))

            with c7:
                st.metric("近 60 日漲跌幅", sa.fmt_pct(metrics["ret60"]))

            with c8:
                st.metric("近 120 日漲跌幅", sa.fmt_pct(metrics["ret120"]))

            price_df = data.get("price_df")

            if isinstance(price_df, pd.DataFrame) and not price_df.empty:
                chart_cols = [
                    col for col in ["Close", "MA60", "MA120", "MA240"]
                    if col in price_df.columns
                ]

                if chart_cols:
                    st.line_chart(price_df[chart_cols].dropna())

                with st.expander("查看最近股價資料"):
                    safe_dataframe(price_df.tail(60))

            st.markdown("### 技術面條件檢查")
            show_details(details["technical"])

        # ==================================================
        # Tab 2：月營收
        # ==================================================

        with tab2:
            st.subheader("月營收成長動能")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("最新營收月份", metrics["revenue_period"])

            with c2:
                st.metric("月營收年增率", sa.fmt_pct(metrics["revenue_yoy"]))

            with c3:
                st.metric("月營收月增率", sa.fmt_pct(metrics["revenue_mom"]))

            with c4:
                st.metric("近 3 個月平均年增率", sa.fmt_pct(metrics["revenue_avg_3m_yoy"]))

            revenue_df = data.get("revenue_df")

            if isinstance(revenue_df, pd.DataFrame) and not revenue_df.empty:
                chart_df = revenue_df[["period", "revenue"]].dropna().copy()
                chart_df = chart_df.set_index("period")

                st.line_chart(chart_df)

                with st.expander("查看最近 24 個月營收資料"):
                    show_cols = [
                        col for col in [
                            "period",
                            "revenue",
                            "mom_pct",
                            "yoy_pct",
                            "avg_3m_yoy",
                            "avg_6m_yoy"
                        ]
                        if col in revenue_df.columns
                    ]

                    safe_dataframe(revenue_df[show_cols].tail(24))

            st.markdown("### 月營收條件檢查")
            show_details(details["revenue"])

        # ==================================================
        # Tab 3：獲利品質
        # ==================================================

        with tab3:
            st.subheader("獲利品質")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("最新 EPS", sa.fmt_num(metrics["eps"]))

            with c2:
                st.metric("EPS 年增率", sa.fmt_pct(metrics["eps_yoy"]))

            with c3:
                st.metric("ROE", sa.fmt_pct(metrics["roe"]))

            with c4:
                st.metric("毛利率", sa.fmt_pct(metrics["gross_margin"]))

            c5, c6 = st.columns(2)

            with c5:
                st.metric("營業利益率", sa.fmt_pct(metrics["operating_margin"]))

            with c6:
                st.metric("近四季 EPS 加總", sa.fmt_num(metrics["ttm_eps"]))

            profit_df = data.get("profit_df")

            if isinstance(profit_df, pd.DataFrame) and not profit_df.empty:
                with st.expander("查看最近 12 季獲利資料"):
                    show_cols = [
                        col for col in [
                            "date",
                            "eps",
                            "eps_yoy_pct",
                            "ttm_eps",
                            "roe",
                            "gross_margin",
                            "operating_margin"
                        ]
                        if col in profit_df.columns
                    ]

                    safe_dataframe(profit_df[show_cols].tail(12))

            st.markdown("### 獲利品質條件檢查")
            show_details(details["profit_quality"])

        # ==================================================
        # Tab 4：估值
        # ==================================================

        with tab4:
            st.subheader("估值合理性")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("本益比 P/E", sa.fmt_num(metrics["pe"]))

            with c2:
                st.metric("P/E 歷史百分位", sa.fmt_pct(metrics["pe_percentile"]))

            with c3:
                st.metric("股價淨值比 P/B", sa.fmt_num(metrics["pb"]))

            with c4:
                st.metric("P/B 歷史百分位", sa.fmt_pct(metrics["pb_percentile"]))

            c5, c6, c7 = st.columns(3)

            with c5:
                st.metric("P/E 近 1 年平均", sa.fmt_num(metrics["pe_1y_avg"]))

            with c6:
                st.metric("P/B 近 1 年平均", sa.fmt_num(metrics["pb_1y_avg"]))

            with c7:
                st.metric("殖利率", sa.fmt_pct(metrics["dividend_yield"]))

            valuation_df = data.get("valuation_df")

            if isinstance(valuation_df, pd.DataFrame) and not valuation_df.empty:
                with st.expander("查看最近 60 筆估值資料"):
                    show_cols = [
                        col for col in [
                            "date",
                            "PER",
                            "PER_percentile",
                            "PBR",
                            "PBR_percentile",
                            "dividend_yield"
                        ]
                        if col in valuation_df.columns
                    ]

                    safe_dataframe(valuation_df[show_cols].tail(60))

            st.markdown("### 估值條件檢查")
            show_details(details["valuation"])

        # ==================================================
        # Tab 5：三大法人
        # ==================================================

        with tab5:
            st.subheader("三大法人籌碼")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("外資當日買賣超", sa.fmt_int(metrics["foreign_net"]))

            with c2:
                st.metric("投信當日買賣超", sa.fmt_int(metrics["investment_trust_net"]))

            with c3:
                st.metric("自營商當日買賣超", sa.fmt_int(metrics["dealer_net"]))

            with c4:
                st.metric("三大法人當日合計", sa.fmt_int(metrics["institutional_total_net"]))

            c5, c6, c7 = st.columns(3)

            with c5:
                st.metric("近 5 日三大法人", sa.fmt_int(metrics["institutional_total_5d"]))

            with c6:
                st.metric("近 20 日三大法人", sa.fmt_int(metrics["institutional_total_20d"]))

            with c7:
                st.metric("近 60 日三大法人", sa.fmt_int(metrics["institutional_total_60d"]))

            institutional_df = data.get("institutional_df")

            if isinstance(institutional_df, pd.DataFrame) and not institutional_df.empty:
                with st.expander("查看最近 60 日法人資料"):
                    show_cols = [
                        col for col in [
                            "date",
                            "foreign_net",
                            "investment_trust_net",
                            "dealer_net",
                            "total_net",
                            "total_5d",
                            "total_20d",
                            "total_60d"
                        ]
                        if col in institutional_df.columns
                    ]

                    safe_dataframe(institutional_df[show_cols].tail(60))

            st.markdown("### 三大法人條件檢查")
            show_details(details["institutional"])

        # ==================================================
        # Tab 6：完整報告
        # ==================================================

        with tab6:
            st.subheader("完整文字報告")

            st.text_area(
                label="分析報告",
                value=report,
                height=700
            )

            st.download_button(
                label="下載分析報告 TXT",
                data=report,
                file_name=f"{stock_id}_stock_analysis_report.txt",
                mime="text/plain"
            )

        # ==================================================
        # 錯誤與缺漏資料
        # ==================================================

        if errors:
            st.markdown("---")
            st.warning("部分資料抓取或分析失敗，可能是資料源缺漏、API 限制或該股票不支援。")

            with st.expander("查看錯誤與資料缺漏原因"):
                for key, value in errors.items():
                    st.write(f"**{key}**：{value}")

    except Exception as e:
        st.error("分析時發生錯誤")
        st.write(str(e))

        with st.expander("查看詳細錯誤"):
            st.code(traceback.format_exc())
else:
    st.info("請輸入股票代號，然後按下「開始分析」。")
