"""
ไฟล์: tests/test_quant_wheel_tools.py
วิธีรัน: pytest tests/test_quant_wheel_tools.py -v
"""

import math
import numpy as np
import pandas as pd
import pytest

from tradingagents.agents import quant_wheel_tools
from tradingagents.agents.quant_wheel_tools import (
    compute_black_scholes_put,
    compute_black_scholes_call,
    select_optimal_covered_call_strike,
    estimate_skewed_iv,
    resolve_optimal_roll_mode,
    evaluate_option_roll_mechanics,
    evaluate_earnings_blackout_from_dates,
    fetch_ticker_earnings_blackout,
    analyze_thai_dividend_and_reit_regime,
    get_weekly_macro_drawdown_regime,
)


def test_black_scholes_put_metrics_and_breakeven():
    res = compute_black_scholes_put(spot=442.10, strike=430.00, dte_days=35, iv_annual=0.24)
    assert res["strike_price"] == 430.00
    assert res["put_premium_per_share"] > 0.0
    assert -1.0 < res["put_delta"] < 0.0
    assert 10.0 < res["prob_assignment_pct"] < 50.0
    expected_be = round(430.00 - res["put_premium_per_share"], 2)
    assert res["net_breakeven_price"] == pytest.approx(expected_be, abs=0.02)


def test_black_scholes_call_metrics_and_exit_price():
    res = compute_black_scholes_call(spot=228.60, strike=245.00, dte_days=35, iv_annual=0.28)
    assert res["call_strike"] == 245.00
    assert res["call_premium_per_share"] > 0.0
    assert 0.0 < res["call_delta"] < 0.50
    expected_exit = round(245.00 + res["call_premium_per_share"], 2)
    assert res["effective_call_away_price"] == pytest.approx(expected_exit, abs=0.02)


def test_put_call_parity_invariant():
    spot, strike, dte, iv, r = 200.0, 195.0, 45, 0.30, 0.0425
    T = dte / 365.0
    call_res = compute_black_scholes_call(spot, strike, dte, iv, r)
    put_res = compute_black_scholes_put(spot, strike, dte, iv, r)
    lhs = call_res["call_premium_per_share"] - put_res["put_premium_per_share"]
    rhs = spot - strike * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=0.03)


def test_select_optimal_covered_call_strike_targets_delta():
    cc = select_optimal_covered_call_strike(spot=615.20, dte_days=35, iv_annual=0.30, target_delta=0.22)
    assert cc["call_strike"] > 615.20
    assert 0.15 <= cc["call_delta"] <= 0.30


def test_volatility_skew_asymmetry():
    base_iv, spot = 0.25, 200.0
    otm_put_iv = estimate_skewed_iv(base_iv, spot=spot, strike=180.0, option_type="PUT")
    atm_iv = estimate_skewed_iv(base_iv, spot=spot, strike=200.0, option_type="PUT")
    otm_call_iv = estimate_skewed_iv(base_iv, spot=spot, strike=220.0, option_type="CALL")
    assert otm_put_iv > atm_iv
    assert otm_call_iv < atm_iv
    assert otm_put_iv == pytest.approx(0.28, abs=1e-4)
    assert otm_call_iv == pytest.approx(0.24, abs=1e-4)


@pytest.mark.parametrize(
    "opt_type,cdc_zone,expected_mode",
    [
        ("CALL", "GREEN", "MAX_STRIKE_SHIFT"),
        ("CALL", "BLUE", "BALANCED_50_50"),
        ("CALL", "YELLOW", "MAX_NET_CREDIT"),
        ("CALL", "RED", "MAX_NET_CREDIT"),
        ("PUT", "RED", "MAX_STRIKE_SHIFT"),
        ("PUT", "YELLOW", "BALANCED_50_50"),
        ("PUT", "BLUE", "MAX_NET_CREDIT"),
        ("PUT", "GREEN", "MAX_NET_CREDIT"),
    ],
)
def test_resolve_optimal_roll_mode_matrix(opt_type, cdc_zone, expected_mode):
    assert resolve_optimal_roll_mode(opt_type, cdc_zone) == expected_mode


def test_call_roll_modes_collapse_when_one_strike_rolls_for_a_credit():
    # A 30-day roll of a challenged 245 call: only the 250 strike still pays a
    # net credit, so every mode lands on it rather than rolling for a debit.
    kw = dict(option_type="CALL", spot=243.50, current_strike=245.0, remaining_dte=21, entry_premium=2.65, iv_annual=0.28, cdc_zone="GREEN")
    plans = {m: evaluate_option_roll_mechanics(**kw, roll_mode=m)["selected_roll_plan"]
             for m in ("MAX_NET_CREDIT", "BALANCED_50_50", "MAX_STRIKE_SHIFT")}
    assert {p["new_strike"] for p in plans.values()} == {250.0}
    assert all(p["net_credit_per_share"] >= 0.0 for p in plans.values())
    assert evaluate_option_roll_mechanics(**kw, roll_mode="AUTO_CDC")["effective_roll_mode"] == "MAX_STRIKE_SHIFT"


def test_call_roll_modes_ordering_and_invariants():
    # A 60-day roll leaves several credit strikes, so the modes separate.
    kw = dict(option_type="CALL", spot=243.50, current_strike=245.0, remaining_dte=21, entry_premium=2.65,
              iv_annual=0.28, cdc_zone="GREEN", roll_add_days=60)
    res_credit = evaluate_option_roll_mechanics(**kw, roll_mode="MAX_NET_CREDIT")
    res_bal = evaluate_option_roll_mechanics(**kw, roll_mode="BALANCED_50_50")
    res_shift = evaluate_option_roll_mechanics(**kw, roll_mode="MAX_STRIKE_SHIFT")
    res_auto = evaluate_option_roll_mechanics(**kw, roll_mode="AUTO_CDC")

    p_c, p_b, p_s = res_credit["selected_roll_plan"], res_bal["selected_roll_plan"], res_shift["selected_roll_plan"]
    assert p_c["new_strike"] <= p_b["new_strike"] <= p_s["new_strike"]
    assert p_s["new_strike"] > p_c["new_strike"]
    assert p_c["net_credit_per_share"] >= p_b["net_credit_per_share"] >= p_s["net_credit_per_share"] >= 0.0
    assert res_auto["effective_roll_mode"] == "MAX_STRIKE_SHIFT"


def test_put_roll_modes_ordering_and_invariants():
    kw = dict(option_type="PUT", spot=429.00, current_strike=430.0, remaining_dte=21, entry_premium=6.45, iv_annual=0.24, cdc_zone="YELLOW")
    res_credit = evaluate_option_roll_mechanics(**kw, roll_mode="MAX_NET_CREDIT")
    res_bal = evaluate_option_roll_mechanics(**kw, roll_mode="BALANCED_50_50")
    res_shift = evaluate_option_roll_mechanics(**kw, roll_mode="MAX_STRIKE_SHIFT")
    res_auto = evaluate_option_roll_mechanics(**kw, roll_mode="AUTO_CDC")

    p_c, p_b, p_s = res_credit["selected_roll_plan"], res_bal["selected_roll_plan"], res_shift["selected_roll_plan"]
    assert p_c["new_strike"] >= p_b["new_strike"] >= p_s["new_strike"]
    assert p_s["new_strike"] < p_c["new_strike"]
    assert p_c["net_credit_per_share"] >= p_b["net_credit_per_share"] >= p_s["net_credit_per_share"] >= 0.0
    assert res_auto["effective_roll_mode"] == "BALANCED_50_50"


def test_take_profit_50_pct_trigger():
    res = evaluate_option_roll_mechanics(
        option_type="PUT", spot=174.00, current_strike=145.0, remaining_dte=21, entry_premium=3.85, iv_annual=0.40
    )
    assert res["profit_capture_pct"] >= 50.0
    assert res["decision_verdict"].startswith("TAKE_PROFIT_50_PCT")


def _weekly(prices, end="2026-11-13"):
    return pd.Series(prices, index=pd.date_range(end=end, periods=len(prices), freq="W-FRI"), dtype=float)


def test_get_weekly_macro_drawdown_regime_tool_with_mock_data(monkeypatch):
    weekly = _weekly(np.linspace(100.0, 200.0, 200).tolist() + [165.0] * 20)
    daily = pd.Series(np.linspace(160.0, 165.0, 60), index=pd.bdate_range(end="2026-11-13", periods=60))
    seen = []
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "weekly_closes", lambda s, d: seen.append((s, d)) or weekly.copy())
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "daily_closes", lambda s, d: daily.copy())
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "earnings_dates", lambda s: ["2026-08-26", "2026-11-19"])

    output = get_weekly_macro_drawdown_regime.invoke(
        {"symbol": "NVDA", "trade_date": "2026-11-13", "benchmark_ticker": "QQQ", "dte_days": 35})
    assert "As of 2026-11-13" in output
    assert "earnings_blackout_gate" in output
    assert "PRE_EARNINGS_BLACKOUT" in output       # NVDA reports inside the 35-day window
    assert "ETF_EXEMPT" in output                  # QQQ has no earnings date
    assert ("NVDA", "2026-11-13") in seen and ("QQQ", "2026-11-13") in seen


def test_weekly_regime_reports_missing_history_instead_of_a_tier(monkeypatch):
    from tradingagents.dataflows.errors import NoMarketDataError
    def boom(s, d):
        raise NoMarketDataError(s, s, "no rows")
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "weekly_closes", boom)
    res = quant_wheel_tools.analyze_weekly_wheel_regime("ZZZZ", "2026-11-13")
    assert "error" in res and "tactical_regime_tier" not in res


def test_thai_listing_gets_the_dividend_regime_not_an_option_leg(monkeypatch):
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "weekly_closes",
                        lambda s, d: _weekly(np.linspace(30.0, 38.0, 160).tolist()))
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "dividends",
                        lambda s: pd.Series([1.0, 1.1], index=pd.to_datetime(["2026-04-20", "2026-08-25"])))
    output = get_weekly_macro_drawdown_regime.invoke({"symbol": "PTT.BK", "trade_date": "2026-11-13"})
    assert "no listed single-stock options" in output
    assert "Tactical Tier" in output and "wheel_option_recommendation" not in output


def test_earnings_blackout_etf_exemption():
    res = evaluate_earnings_blackout_from_dates("QQQ", "2026-11-13", ["2026-11-19"], dte_days=35)
    assert res["is_etf_exempt"] is True
    assert res["is_blackout"] is False
    assert res["blackout_status"] == "ETF_EXEMPT"


def test_pre_earnings_blackout_blocks_nvda_short_option():
    res = evaluate_earnings_blackout_from_dates(
        symbol="NVDA",
        trade_date="2026-11-13",
        earnings_dates=["2026-08-26", "2026-11-19", "2027-02-24"],
        dte_days=35,
    )
    assert res["is_blackout"] is True
    assert res["blackout_status"] == "PRE_EARNINGS_BLACKOUT"
    assert res["days_to_next_earnings"] == 6
    assert res["safe_resume_date"] == "2026-11-23"


def test_post_earnings_3day_cooldown_blocks_immediate_entry():
    res = evaluate_earnings_blackout_from_dates(
        symbol="NVDA",
        trade_date="2026-11-20",
        earnings_dates=["2026-11-19", "2027-02-24"],
        dte_days=35,
    )
    assert res["is_blackout"] is True
    assert res["blackout_status"] == "POST_EARNINGS_COOLDOWN"
    assert res["days_since_last_earnings"] == 1


def test_clear_earnings_window_allows_normal_wheel():
    res = evaluate_earnings_blackout_from_dates(
        symbol="MSFT",
        trade_date="2026-11-13",
        earnings_dates=["2026-10-28", "2027-01-27"],
        dte_days=35,
    )
    assert res["is_blackout"] is False
    assert res["blackout_status"] == "CLEAR"
    assert res["days_to_next_earnings"] == 75


def test_roll_mechanics_forces_close_before_imminent_earnings():
    res = evaluate_option_roll_mechanics(
        option_type="PUT",
        spot=162.85,
        current_strike=145.0,
        remaining_dte=14,
        entry_premium=3.85,
        iv_annual=0.42,
        days_to_next_earnings=4,
    )
    assert res["decision_verdict"].startswith("CLOSE_BEFORE_EARNINGS")


def test_fetch_ticker_earnings_blackout_with_mock_data(monkeypatch):
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "earnings_dates",
                        lambda s: ["2026-08-26", "2026-11-19", "2027-02-24"])
    res = fetch_ticker_earnings_blackout("NVDA", trade_date="2026-11-13", dte_days=35)
    assert res["is_blackout"] is True
    assert res["blackout_status"] == "PRE_EARNINGS_BLACKOUT"


def test_unreadable_earnings_calendar_blocks_instead_of_clearing(monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError
    def boom(s):
        raise VendorUnavailableError("Yahoo Finance request failed")
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "earnings_dates", boom)
    res = fetch_ticker_earnings_blackout("NVDA", trade_date="2026-11-13", dte_days=35)
    assert res["is_blackout"] is True
    assert res["blackout_status"] == "EARNINGS_DATES_UNAVAILABLE"


def test_no_listed_earnings_dates_is_not_clear():
    res = evaluate_earnings_blackout_from_dates("MSFT", "2026-11-13", [], dte_days=35)
    assert res["is_blackout"] is True
    assert res["blackout_status"] == "EARNINGS_DATES_UNAVAILABLE"


def test_analyze_thai_dividend_and_reit_regime_with_mock_data(monkeypatch):
    prices = np.linspace(145.0, 160.0, 130).tolist() + np.linspace(155.0, 124.50, 30).tolist()
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "weekly_closes", lambda s, d: _weekly(prices))
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "dividends",
                        lambda s: pd.Series([5.25, 4.25], index=pd.to_datetime(["2026-03-01", "2026-08-15"])))

    res = analyze_thai_dividend_and_reit_regime(
        symbol="PTTEP",
        trade_date="2026-11-13",
        fair_value_estimate=148.00,
    )
    assert res["Ticker"] == "PTTEP.BK"
    assert res["Spot"] == 124.50
    assert res["TTM Div (฿/sh)"] == 9.50
    assert res["Gross Yield (%)"] == pytest.approx(7.63, abs=0.05)
    assert res["Net Yield 90% (%)"] == pytest.approx(6.87, abs=0.05)
    assert res["10% MoS Entry (฿)"] == 133.20
    assert res["Tactical Tier"] in ("TIER_2_ACCUMULATION", "TIER_3_DEEP_VALUE")
    # 160 weeks is short of the 200 an EMA200 needs: reported as missing, not estimated.
    assert res["Weekly EMA200"] is None


def test_dividends_after_the_analysis_date_are_not_counted(monkeypatch):
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "weekly_closes",
                        lambda s, d: _weekly(np.linspace(30.0, 38.0, 160).tolist(), end="2026-05-01"))
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "dividends",
                        lambda s: pd.Series([1.0, 1.1], index=pd.to_datetime(["2026-04-20", "2026-08-25"])))
    res = analyze_thai_dividend_and_reit_regime("PTT.BK", "2026-05-01")
    assert res["TTM Div (฿/sh)"] == 1.0
    assert res["Last XD Date"] == "2026-04-20"


# --- the wheel decision: tiers, holdings, CDC zone, delta target, open options ---

def _trend(spot, ema100, ema200=None, dd=-5.0, cdc="YELLOW"):
    return {"spot": spot, "ema12": spot, "ema26": spot, "ema100": ema100, "ema200": ema200,
            "peak_2yr": spot / (1 + dd / 100), "dd_pct": dd,
            "dist_ema100_pct": (spot / ema100 - 1) * 100, "cdc_zone": cdc}


@pytest.fixture
def market(monkeypatch):
    """Scripted weekly trend and a clear earnings calendar; returns a setter for the trend."""
    daily = pd.Series(100 * np.exp(np.cumsum(np.full(60, 0.0)) + np.sin(np.arange(60)) * 0.01),
                      index=pd.bdate_range(end="2026-11-13", periods=60))
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "weekly_closes", lambda s, d: _weekly([100.0] * 150))
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "daily_closes", lambda s, d: daily)
    monkeypatch.setattr(quant_wheel_tools.wheel_data, "earnings_dates", lambda s: ["2026-10-28", "2027-01-27"])
    state = {}
    monkeypatch.setattr(quant_wheel_tools, "_weekly_trend", lambda close: state["trend"])

    def set_trend(**kw):
        state["trend"] = _trend(**kw)
    return set_trend


def _run(**kw):
    return quant_wheel_tools.analyze_weekly_wheel_regime("MSFT", "2026-11-13", 35, **kw)


def test_tier3_buys_spot_and_offers_no_option(market):
    market(spot=100.0, ema100=130.0, ema200=110.0, dd=-40.0, cdc="RED")
    res = _run()
    assert res["tactical_regime_tier"] == "TIER_3_DEEP_VALUE"
    assert res["wheel_option_recommendation"] is None
    assert res["execution_vehicle"] == "EQUITY_SPOT"


def test_drawdown_sets_tier2_above_the_ema100_and_says_so(market):
    market(spot=105.3, ema100=100.0, dd=-26.0, cdc="YELLOW")
    res = _run()
    assert res["tactical_regime_tier"] == "TIER_2_ACCUMULATION"
    assert "overrides EMA100 distance" in res["tier_reason"]


def test_tier2_put_is_deeper_in_a_red_weekly_trend(market):
    market(spot=100.0, ema100=105.0, dd=-10.0, cdc="YELLOW")
    normal = _run()["wheel_option_recommendation"]["strike_price"]
    market(spot=100.0, ema100=105.0, dd=-10.0, cdc="RED")
    red = _run()["wheel_option_recommendation"]["strike_price"]
    assert normal == 90.0 and red == 85.0


def test_tier1_waits_instead_of_selling_a_put_in_a_red_weekly_trend(market):
    market(spot=103.0, ema100=100.0, dd=-8.0, cdc="RED")
    res = _run()
    assert res["tactical_regime_tier"] == "TIER_1_WATCH_ZONE"
    assert res["wheel_option_recommendation"] is None and res["wheel_action"].startswith("WAIT")
    market(spot=103.0, ema100=100.0, dd=-8.0, cdc="YELLOW")
    assert _run()["execution_vehicle"] == "CASH_SECURED_PUT"


@pytest.mark.parametrize("shares,vehicle,phrase", [
    (None, "COVERED_CALL_OVERWRITE", "only if 100+ shares"),
    (50, "NONE", "NO NEW COVERED CALL (holds 50 shares"),
    (250, "COVERED_CALL_OVERWRITE", "SELL 2 COVERED CALL"),
])
def test_covered_call_depends_on_the_shares_held(market, shares, vehicle, phrase):
    market(spot=120.0, ema100=100.0, dd=-2.0, cdc="YELLOW")
    res = _run(shares_held=shares)
    assert res["tactical_regime_tier"] == "TIER_0_EXTENDED"
    assert res["execution_vehicle"] == vehicle and phrase in res["wheel_action"]
    if shares is None:
        assert any("naked call" in n for n in res["notes"])


def test_green_weekly_trend_targets_a_further_call(market):
    market(spot=120.0, ema100=100.0, dd=-2.0, cdc="GREEN")
    green = _run(shares_held=100)["wheel_option_recommendation"]
    market(spot=120.0, ema100=100.0, dd=-2.0, cdc="YELLOW")
    other = _run(shares_held=100)["wheel_option_recommendation"]
    assert green["target_delta"] == 0.15 and other["target_delta"] == 0.24
    assert green["call_strike"] > other["call_strike"]


def test_an_unreachable_delta_target_is_flagged():
    cc = select_optimal_covered_call_strike(spot=100.0, dte_days=35, iv_annual=4.0, target_delta=0.15)
    assert cc["delta_off_target"] is True
    assert select_optimal_covered_call_strike(100.0, 35, 0.30, 0.22)["delta_off_target"] is False


def test_open_short_options_get_a_roll_verdict(market):
    market(spot=120.0, ema100=100.0, dd=-2.0, cdc="GREEN")
    res = _run(shares_held=100, open_options=[
        {"option_type": "CALL", "strike": 120.0, "expiry": "2026-11-27", "contracts": 1, "entry_premium": 2.0},
        {"option_type": "PUT", "strike": 90.0, "expiry": "2026-11-13", "contracts": 1, "entry_premium": 1.0},
    ])
    challenged, expired = res["open_option_reviews"]
    assert challenged["remaining_dte"] == 14
    assert challenged["decision_verdict"].startswith("ROLL_UP")
    assert challenged["effective_roll_mode"] == "MAX_STRIKE_SHIFT"   # GREEN call
    assert expired["decision_verdict"].startswith("EXPIRED")


def test_tool_reads_holdings_and_open_options_from_the_portfolio(market):
    market(spot=120.0, ema100=100.0, dd=-2.0, cdc="YELLOW")
    book = {"positions": [{"ticker": "MSFT", "quantity": 40}],
            "options": [{"ticker": "MSFT", "option_type": "put", "strike": 100, "expiry": "2026-12-18",
                         "contracts": 1, "entry_premium": 2.5}]}
    out = get_weekly_macro_drawdown_regime.invoke(
        {"symbol": "MSFT", "trade_date": "2026-11-13", "portfolio_book": book})
    assert "NO NEW COVERED CALL (holds 40 shares" in out
    assert "open_option_reviews" in out
    out_no_book = get_weekly_macro_drawdown_regime.invoke({"symbol": "MSFT", "trade_date": "2026-11-13", "portfolio_book": {}})
    assert "only if 100+ shares" in out_no_book and "open_option_reviews" not in out_no_book


def test_shares_under_an_open_call_are_not_offered_again(market):
    market(spot=120.0, ema100=100.0, dd=-2.0, cdc="YELLOW")
    open_call = {"option_type": "CALL", "strike": 130.0, "expiry": "2026-12-18", "contracts": 1, "entry_premium": 2.0}
    res = _run(shares_held=100, open_options=[open_call])
    assert res["shares_free_for_calls"] == 0
    assert res["wheel_action"].startswith("NO NEW COVERED CALL") and "100 already under open calls" in res["wheel_action"]
    assert "SELL 1 COVERED CALL" in _run(shares_held=200, open_options=[open_call])["wheel_action"]


def test_a_new_csp_must_fit_in_the_free_cash(market):
    market(spot=103.0, ema100=100.0, dd=-8.0, cdc="YELLOW")
    res = _run(cash_available=5_000.0)
    assert res["wheel_action"].startswith("WAIT (CSP needs 10,000 cash per contract, 5,000 free")
    assert _run(cash_available=20_000.0)["execution_vehicle"] == "CASH_SECURED_PUT"
    assert _run()["execution_vehicle"] == "CASH_SECURED_PUT"          # cash unknown: not blocked


def test_free_cash_subtracts_open_put_collateral():
    from tradingagents.portfolio import PortfolioContext
    book = PortfolioContext.model_validate({"cash": 50_000, "options": [
        {"ticker": "META", "option_type": "PUT", "strike": 400, "expiry": "2026-12-04", "contracts": 1, "entry_premium": 9}]})
    assert quant_wheel_tools.free_cash(book) == 10_000
    assert quant_wheel_tools.free_cash(PortfolioContext()) is None
