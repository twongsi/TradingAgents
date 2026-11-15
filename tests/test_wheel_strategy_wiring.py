"""The options-wheel mode is opt-in: off, runs are unchanged; on, the analyst and trader get it."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.agents.analysts import market_analyst
from tradingagents.agents.schemas import TraderAction, TraderProposal, render_trader_proposal
from tradingagents.agents.trader import trader
from tradingagents.dataflows.vendors.yahoo import wheel
from tradingagents.default_config import build_default_config


def _config(monkeypatch, module, on):
    monkeypatch.setattr(module, "get_config", lambda: {"wheel_strategy": on})


@pytest.mark.unit
def test_wheel_mode_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_WHEEL_STRATEGY", raising=False)
    assert build_default_config()["wheel_strategy"] is False


@pytest.mark.unit
def test_wheel_mode_reads_its_environment_variable(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_WHEEL_STRATEGY", "true")
    assert build_default_config()["wheel_strategy"] is True


@pytest.mark.unit
@pytest.mark.parametrize("on", [False, True])
def test_market_analyst_offers_the_wheel_tool_only_in_wheel_mode(monkeypatch, on):
    _config(monkeypatch, market_analyst, on)
    names = {t.name for t in market_analyst.offered_tools()}
    assert ("get_weekly_macro_drawdown_regime" in names) is on
    # The tool node can always run it, whichever tools a run offers.
    assert "get_weekly_macro_drawdown_regime" in {t.name for t in market_analyst.TOOLS}


@pytest.mark.unit
@pytest.mark.parametrize("on", [False, True])
def test_trader_gets_the_wheel_rules_only_in_wheel_mode(monkeypatch, on):
    _config(monkeypatch, trader, on)
    assert ("EARNINGS BLACKOUT" in trader.wheel_mandate()) is on


@pytest.mark.unit
def test_wheel_fields_render_only_when_filled():
    plain = render_trader_proposal(TraderProposal(action=TraderAction.HOLD, reasoning="r"))
    assert "Macro Regime Tier" not in plain
    wheel_plan = render_trader_proposal(TraderProposal(
        action=TraderAction.BUY, reasoning="r", macro_regime_tier="TIER_1_WATCH_ZONE",
        execution_vehicle="CASH_SECURED_PUT", earnings_blackout_status="CLEAR"))
    assert "**Macro Regime Tier**: TIER_1_WATCH_ZONE" in wheel_plan
    assert "**Execution Vehicle**: CASH_SECURED_PUT" in wheel_plan
    assert wheel_plan.rstrip().endswith("FINAL TRANSACTION PROPOSAL: **BUY**")


@pytest.mark.unit
def test_weekly_closes_are_friday_closes_cut_at_the_analysis_date(monkeypatch):
    days = pd.bdate_range("2026-10-26", "2026-11-11")  # Mon .. Wed
    frame = pd.DataFrame({"Date": days, "Close": range(len(days))})
    monkeypatch.setattr(wheel, "load_ohlcv", lambda s, d: frame[frame["Date"] <= d].copy())
    weekly = wheel.weekly_closes("PTT.BK", "2026-11-11")
    assert list(weekly.index.strftime("%Y-%m-%d")) == ["2026-10-30", "2026-11-06", "2026-11-13"]
    assert weekly.iloc[-1] == len(days) - 1   # the partial week ends at Wednesday's close


@pytest.mark.unit
def test_dividends_index_is_timezone_naive(monkeypatch):
    idx = pd.DatetimeIndex(["2026-04-20 00:00:00+07:00"])
    monkeypatch.setattr(wheel.yf, "Ticker", lambda s: SimpleNamespace(dividends=pd.Series([1.0], index=idx)))
    out = wheel.dividends("PTT.BK")
    assert out.index.tz is None and out.index[0] == pd.Timestamp("2026-04-20")


@pytest.mark.unit
def test_earnings_dates_are_sorted_unique_days(monkeypatch):
    idx = pd.DatetimeIndex(["2026-11-19 16:00", "2026-08-26 16:00", "2026-11-19 16:00"])
    monkeypatch.setattr(wheel.yf, "Ticker", lambda s: SimpleNamespace(
        get_earnings_dates=lambda limit=12: pd.DataFrame({"EPS": [1, 2, 3]}, index=idx)))
    assert wheel.earnings_dates("NVDA") == ["2026-08-26", "2026-11-19"]
