#!/usr/bin/env python3
"""
ไฟล์: backtest_nvda_2022_2026.py
วิธีรัน: python backtest_nvda_2022_2026.py
"""

import numpy as np
import pandas as pd
import yfinance as yf


def run_vectorized_portfolio_comparison(
    ticker: str = "NVDA",
    start_date: str = "2022-01-03",
    end_date: str = "2026-11-13",
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    warmup_start = "2018-01-01"
    df_nvda = yf.download(ticker, start=warmup_start, end=end_date, interval="1wk", progress=False)
    if isinstance(df_nvda.columns, pd.MultiIndex):
        df_nvda.columns = df_nvda.columns.get_level_values(0)

    data = pd.DataFrame(index=df_nvda.index)
    data["close"] = df_nvda["Close"]
    data["ema12"] = data["close"].ewm(span=12, adjust=False).mean()
    data["ema26"] = data["close"].ewm(span=26, adjust=False).mean()
    data["ema100"] = data["close"].ewm(span=100, adjust=False).mean()
    data["ema200"] = data["close"].ewm(span=200, adjust=False).mean()
    data["peak_104w"] = data["close"].rolling(window=104, min_periods=1).max()
    data["drawdown_pct"] = ((data["close"] - data["peak_104w"]) / data["peak_104w"]) * 100.0

    bt = data.loc[start_date:end_date].copy()
    first_price = float(bt["close"].iloc[0])
    bt["equity_buy_hold"] = (initial_capital / first_price) * bt["close"]

    std_cash, std_shares, std_eq = initial_capital, 0.0, []
    for _, row in bt.iterrows():
        p = float(row["close"])
        if row["ema12"] > row["ema26"] and p > row["ema12"] and std_cash > 0:
            deploy = std_cash * 0.85
            std_shares += deploy / p
            std_cash -= deploy
        elif row["ema12"] < row["ema26"] and p < row["ema26"] * 0.95 and std_shares > 0:
            liq = std_shares * 0.50
            std_cash += liq * p
            std_shares -= liq
        std_eq.append(std_cash + std_shares * p)
    bt["equity_standard_agent"] = std_eq

    c_cash = initial_capital * 0.75
    c_shares = (initial_capital * 0.25) / first_price
    c_eq, t1_used, t2_used = [], False, False
    for _, row in bt.iterrows():
        p, dd, e100, e200 = float(row["close"]), float(row["drawdown_pct"]), float(row["ema100"]), float(row["ema200"])
        if (p <= e100 or dd <= -25.0) and not t1_used and c_cash > 5000:
            dep = c_cash * 0.50
            c_shares += dep / p
            c_cash -= dep
            t1_used = True
        if (p <= e200 or dd <= -35.0) and not t2_used and c_cash > 1000:
            dep = c_cash * 0.95
            c_shares += dep / p
            c_cash -= dep
            t2_used = True
        if p >= e100 * 1.55 and c_shares > 0:
            tot = c_cash + c_shares * p
            if (c_cash / tot) < 0.25:
                trim = c_shares * 0.15
                c_cash += trim * p
                c_shares -= trim
                t1_used, t2_used = False, False
        c_eq.append(c_cash + c_shares * p)
    bt["equity_custom_ema_agent"] = c_eq
    return bt


if __name__ == "__main__":
    result = run_vectorized_portfolio_comparison()
    last = result.iloc[-1]
    print(f"Backtest {result.index[0].date()} -> {result.index[-1].date()} (weekly, Yahoo adjusted closes)")
    for col in ("equity_buy_hold", "equity_standard_agent", "equity_custom_ema_agent"):
        print(f"  {col:26s} {last[col]:>14,.0f}")
    print("Past results do not predict future returns; research only.")
