"""Yahoo data for the options-wheel tools: daily and weekly closes, earnings dates, dividends.

Prices come from ``load_ohlcv``, so they are as traded and cut at the analysis
date. Failures raise ``VendorError`` subclasses for the caller to report.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv


def daily_closes(symbol: str, as_of_date: str) -> pd.Series:
    """Daily closes on or before ``as_of_date``, indexed by session date."""
    data = load_ohlcv(symbol, as_of_date)
    return pd.Series(
        data["Close"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(data["Date"]),
        name="Close",
    )


def weekly_closes(symbol: str, as_of_date: str) -> pd.Series:
    """Weekly (Friday) closes on or before ``as_of_date``; the last week may be partial."""
    return daily_closes(symbol, as_of_date).resample("W-FRI").last().dropna()


def earnings_dates(symbol: str, limit: int = 12) -> list[str]:
    """Past and scheduled earnings dates as YYYY-MM-DD, as Yahoo lists them."""
    frame = yf_retry(lambda: yf.Ticker(normalize_symbol(symbol)).get_earnings_dates(limit=limit))
    if frame is None or frame.empty:
        return []
    return sorted({pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in frame.index})


def dividends(symbol: str) -> pd.Series:
    """Cash dividends per share by ex-date (timezone-naive)."""
    series = yf_retry(lambda: yf.Ticker(normalize_symbol(symbol)).dividends)
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return pd.Series(series.to_numpy(dtype=float), index=index.normalize())
