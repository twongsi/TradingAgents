"""Prices shown to the agents are as traded, not dividend-adjusted.

A dividend-adjusted series rewrites every earlier bar for dividends paid later,
so a backtest dated before an ex-dividend date would quote prices that never
traded. Settlement keeps its own adjusted closes for total return.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.yahoo import market, ohlcv


def _frame():
    return pd.DataFrame(
        {"Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0],
         "Close": [100.5, 101.5], "Adj Close": [95.0, 96.0], "Volume": [1, 1]},
        index=pd.DatetimeIndex(["2026-05-07", "2026-05-08"], name="Date"),
    )


@pytest.mark.unit
def test_load_ohlcv_downloads_unadjusted_prices(monkeypatch, tmp_path):
    calls = []

    def history(**kwargs):
        calls.append(kwargs)
        return _frame()

    monkeypatch.setattr(ohlcv, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(ohlcv.pd.Timestamp, "today", staticmethod(lambda: pd.Timestamp("2026-05-08")))
    monkeypatch.setattr(ohlcv.yf, "Ticker", lambda symbol: SimpleNamespace(history=history))

    out = ohlcv.load_ohlcv("PTT.BK", "2026-05-08")

    assert calls and calls[0]["auto_adjust"] is False
    assert out["Close"].iloc[-1] == 101.5


@pytest.mark.unit
def test_stock_data_tool_downloads_unadjusted_prices(monkeypatch):
    calls = []

    def history(**kwargs):
        calls.append(kwargs)
        return _frame()

    monkeypatch.setattr(market.yf, "Ticker", lambda symbol: SimpleNamespace(history=history))

    text = market.get_YFin_data_online("PTT.BK", "2026-05-07", "2026-05-08")

    assert calls and calls[0]["auto_adjust"] is False
    assert "101.5" in text
