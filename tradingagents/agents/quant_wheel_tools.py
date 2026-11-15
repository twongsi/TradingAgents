"""
โมดูล: tradingagents/agents/quant_wheel_tools.py
รวมฟังก์ชัน:
- Weekly EMA100/200 Drawdown Gate & CDC Action Zone
- Black-Scholes Put/Call & Volatility Skew (Put Smirk / Call Skew)
- 3-Mode Roll Mechanics (MAX_STRIKE_SHIFT, BALANCED_50_50, MAX_NET_CREDIT, AUTO_CDC)
- Earnings Date Blackout Gate & Post-Earnings 3-Day Settlement Cooldown
- Thai Dividend Equity & REIT/IFF Scanner (.BK) with 10% Margin of Safety & 10% Tax Deduction
"""

import math
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.yahoo import wheel as wheel_data

RollModeType = Literal["AUTO_CDC", "MAX_STRIKE_SHIFT", "BALANCED_50_50", "MAX_NET_CREDIT"]
ETF_EXEMPT_SYMBOLS = {"QQQ", "SPY", "IWM", "DIA", "SOXX", "SMH", "XLK", "SGOV", "TLT", "GLD"}
THAI_REIT_AND_IFF_SYMBOLS = {
    "CPNREIT.BK", "FTREIT.BK", "WHAIR.BK", "WHART.BK",
    "DIF.BK", "3BBIF.BK", "ALLY.BK", "LHHOTEL.BK", "IMPACT.BK"
}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _day(value: str) -> str:
    """A date in canonical YYYY-MM-DD form; empty means today (a call outside a run)."""
    return (pd.Timestamp(value) if value else pd.Timestamp.today()).strftime("%Y-%m-%d")


def _cdc_zone(ema12: float, ema26: float, spot: float) -> str:
    """CDC Action Zone from the weekly EMA12/EMA26 cross and the close."""
    if ema12 > ema26:
        return "GREEN" if spot > ema12 else "YELLOW"
    return "BLUE" if spot > ema12 else "RED"


def _weekly_trend(close: pd.Series) -> dict:
    """Weekly EMAs, CDC zone and 2-year drawdown of a weekly close series.

    EMA200 is reported only with at least 200 weeks of history; a shorter
    series leaves it None rather than inventing a level.
    """
    spot = float(close.iloc[-1])
    ema12 = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
    ema26 = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
    ema100 = float(close.ewm(span=100, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else None
    peak_2yr = float(close.rolling(104, min_periods=1).max().iloc[-1])
    return {
        "spot": spot,
        "ema12": ema12,
        "ema26": ema26,
        "ema100": ema100,
        "ema200": ema200,
        "peak_2yr": peak_2yr,
        "dd_pct": ((spot - peak_2yr) / peak_2yr) * 100.0,
        "dist_ema100_pct": ((spot / ema100) - 1.0) * 100.0,
        "cdc_zone": _cdc_zone(ema12, ema26, spot),
    }


def estimate_skewed_iv(
    base_atm_iv: float,
    spot: float,
    strike: float,
    option_type: Literal["CALL", "PUT"],
) -> float:
    """ปรับค่า Implied Volatility ตามระยะห่าง Moneyness (Put Smirk & Call Skew)"""
    moneyness_dist = (strike - spot) / spot
    if option_type == "PUT" and moneyness_dist < 0:
        skew_multiplier = 1.0 + abs(moneyness_dist) * 1.20
    elif option_type == "CALL" and moneyness_dist > 0:
        skew_multiplier = max(0.85, 1.0 - moneyness_dist * 0.40)
    else:
        skew_multiplier = 1.0 + abs(moneyness_dist) * 0.30
    return round(max(0.10, base_atm_iv * skew_multiplier), 4)


def compute_black_scholes_put(
    spot: float,
    strike: float,
    dte_days: int,
    iv_annual: float,
    risk_free_rate: float = 0.0425,
) -> dict:
    T = max(dte_days / 365.0, 1e-5)
    sigma = max(iv_annual, 0.10)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put_price = strike * math.exp(-risk_free_rate * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    put_delta = _norm_cdf(d1) - 1.0
    prob_itm_pct = _norm_cdf(-d2) * 100.0

    net_breakeven = strike - put_price
    period_return_pct = (put_price / strike) * 100.0
    annualized_yield_pct = period_return_pct * (365.0 / dte_days)

    return {
        "strike_price": round(strike, 2),
        "dte_days": dte_days,
        "estimated_iv_pct": round(sigma * 100.0, 2),
        "put_premium_per_share": round(put_price, 2),
        "total_premium_1_contract": round(put_price * 100.0, 2),
        "cash_collateral_1_contract": round(strike * 100.0, 2),
        "net_breakeven_price": round(net_breakeven, 2),
        "discount_from_spot_pct": round(((net_breakeven / spot) - 1.0) * 100.0, 2),
        "put_delta": round(put_delta, 3),
        "prob_assignment_pct": round(prob_itm_pct, 1),
        "period_yield_pct": round(period_return_pct, 2),
        "annualized_yield_pct": round(annualized_yield_pct, 2),
    }


def compute_black_scholes_call(
    spot: float,
    strike: float,
    dte_days: int,
    iv_annual: float,
    risk_free_rate: float = 0.0425,
) -> dict:
    T = max(dte_days / 365.0, 1e-5)
    sigma = max(iv_annual, 0.10)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * T) * _norm_cdf(d2)
    call_delta = _norm_cdf(d1)
    prob_called_away_pct = _norm_cdf(d2) * 100.0

    effective_exit_price = strike + call_price
    upside_cap_pct = ((effective_exit_price / spot) - 1.0) * 100.0
    period_yield_pct = (call_price / spot) * 100.0
    annualized_yield_pct = period_yield_pct * (365.0 / dte_days)

    return {
        "call_strike": round(strike, 2),
        "dte_days": dte_days,
        "estimated_iv_pct": round(sigma * 100.0, 2),
        "call_premium_per_share": round(call_price, 2),
        "total_premium_1_contract": round(call_price * 100.0, 2),
        "effective_call_away_price": round(effective_exit_price, 2),
        "max_upside_capture_pct": round(upside_cap_pct, 2),
        "call_delta": round(call_delta, 3),
        "prob_called_away_pct": round(prob_called_away_pct, 1),
        "period_yield_pct": round(period_yield_pct, 2),
        "annualized_yield_pct": round(annualized_yield_pct, 2),
    }


DELTA_TOLERANCE = 0.05


def select_optimal_covered_call_strike(
    spot: float,
    dte_days: int,
    iv_annual: float,
    target_delta: float = 0.22,
    strike_step: float = 5.0,
    max_strike_mult: float = 1.60,
) -> dict:
    """The call strike whose delta is closest to ``target_delta``.

    Strikes from 3% to ``max_strike_mult`` above spot are tried. When none comes
    within DELTA_TOLERANCE of the target (a very volatile stock), the closest is
    returned with ``delta_off_target`` set, so the caller can say so.
    """
    best_eval = None
    min_diff = 999.0
    seen = set()
    for mult in np.linspace(1.03, max_strike_mult, 120):
        candidate_strike = math.ceil((spot * mult) / strike_step) * strike_step
        if candidate_strike in seen:
            continue
        seen.add(candidate_strike)
        res = compute_black_scholes_call(spot, candidate_strike, dte_days, iv_annual)
        diff = abs(res["call_delta"] - target_delta)
        if diff < min_diff:
            min_diff = diff
            best_eval = res
    best_eval = dict(best_eval)
    best_eval["target_delta"] = target_delta
    best_eval["delta_off_target"] = min_diff > DELTA_TOLERANCE
    return best_eval


def resolve_optimal_roll_mode(
    option_type: Literal["CALL", "PUT"],
    cdc_zone: Literal["GREEN", "YELLOW", "BLUE", "RED"] = "YELLOW",
) -> Literal["MAX_STRIKE_SHIFT", "BALANCED_50_50", "MAX_NET_CREDIT"]:
    if option_type == "CALL":
        if cdc_zone == "GREEN":
            return "MAX_STRIKE_SHIFT"
        elif cdc_zone == "BLUE":
            return "BALANCED_50_50"
        return "MAX_NET_CREDIT"
    else:
        if cdc_zone == "RED":
            return "MAX_STRIKE_SHIFT"
        elif cdc_zone == "YELLOW":
            return "BALANCED_50_50"
        return "MAX_NET_CREDIT"


def evaluate_earnings_blackout_from_dates(
    symbol: str,
    trade_date: str,
    earnings_dates: list[str],
    dte_days: int = 35,
    post_earnings_cooldown_days: int = 3,
) -> dict:
    sym_upper = symbol.upper().strip()
    if sym_upper in ETF_EXEMPT_SYMBOLS:
        return {
            "symbol": sym_upper,
            "is_etf_exempt": True,
            "is_blackout": False,
            "blackout_status": "ETF_EXEMPT",
            "last_earnings_date": None,
            "days_since_last_earnings": None,
            "next_earnings_date": None,
            "days_to_next_earnings": None,
            "safe_resume_date": trade_date,
            "action_override": "PROCEED_NORMAL_WHEEL",
            "earnings_note": f"{sym_upper} เป็นกองทุน ETF ไม่มีวันประกาศงบรายบริษัท",
        }

    ref_dt = pd.to_datetime(trade_date).normalize()
    parsed_dates = sorted(
        {pd.to_datetime(d).tz_localize(None).normalize() for d in earnings_dates if d is not None}
    )

    # No dates at all means the calendar could not be read, not that the window
    # is clear: selling an option across an unseen report is the risk this gate
    # exists to stop, so it blocks until a date is confirmed.
    if not parsed_dates:
        return {
            "symbol": sym_upper,
            "is_etf_exempt": False,
            "is_blackout": True,
            "blackout_status": "EARNINGS_DATES_UNAVAILABLE",
            "last_earnings_date": None,
            "days_since_last_earnings": None,
            "next_earnings_date": None,
            "days_to_next_earnings": None,
            "safe_resume_date": None,
            "action_override": "VERIFY_EARNINGS_DATE_BEFORE_SHORT_OPTIONS",
            "earnings_note": f"UNKNOWN: ไม่พบวันประกาศงบของ {sym_upper} ต้องตรวจสอบเองก่อนขายออปชัน",
        }

    past_dates = [d for d in parsed_dates if d < ref_dt]
    future_dates = [d for d in parsed_dates if d >= ref_dt]

    last_dt = past_dates[-1] if past_dates else None
    next_dt = future_dates[0] if future_dates else None

    days_since_last = int((ref_dt - last_dt).days) if last_dt is not None else None
    days_to_next = int((next_dt - ref_dt).days) if next_dt is not None else None

    if days_to_next is not None and 0 <= days_to_next <= dte_days:
        next_str = next_dt.strftime("%Y-%m-%d")
        resume_str = (next_dt + pd.Timedelta(days=post_earnings_cooldown_days + 1)).strftime("%Y-%m-%d")
        return {
            "symbol": sym_upper,
            "is_etf_exempt": False,
            "is_blackout": True,
            "blackout_status": "PRE_EARNINGS_BLACKOUT",
            "last_earnings_date": last_dt.strftime("%Y-%m-%d") if last_dt is not None else None,
            "days_since_last_earnings": days_since_last,
            "next_earnings_date": next_str,
            "days_to_next_earnings": days_to_next,
            "safe_resume_date": resume_str,
            "action_override": "BLOCK_SHORT_OPTION_PARK_IN_SGOV",
            "earnings_note": f"EARNINGS BLACKOUT: งบออก {next_str} (อีก {days_to_next} วัน) -> พักใน SGOV รอวันที่ {resume_str}",
        }

    if days_since_last is not None and 0 <= days_since_last < post_earnings_cooldown_days:
        last_str = last_dt.strftime("%Y-%m-%d")
        resume_str = (last_dt + pd.Timedelta(days=post_earnings_cooldown_days + 1)).strftime("%Y-%m-%d")
        return {
            "symbol": sym_upper,
            "is_etf_exempt": False,
            "is_blackout": True,
            "blackout_status": "POST_EARNINGS_COOLDOWN",
            "last_earnings_date": last_str,
            "days_since_last_earnings": days_since_last,
            "next_earnings_date": next_dt.strftime("%Y-%m-%d") if next_dt is not None else None,
            "days_to_next_earnings": days_to_next,
            "safe_resume_date": resume_str,
            "action_override": "WAIT_3_DAY_POST_EARNINGS_SETTLEMENT",
            "earnings_note": f"POST-EARNINGS COOLDOWN: เพิ่งประกาศงบเมื่อ {last_str} -> รอวันที่ {resume_str}",
        }

    return {
        "symbol": sym_upper,
        "is_etf_exempt": False,
        "is_blackout": False,
        "blackout_status": "CLEAR",
        "last_earnings_date": last_dt.strftime("%Y-%m-%d") if last_dt is not None else None,
        "days_since_last_earnings": days_since_last,
        "next_earnings_date": next_dt.strftime("%Y-%m-%d") if next_dt is not None else None,
        "days_to_next_earnings": days_to_next,
        "safe_resume_date": trade_date,
        "action_override": "PROCEED_NORMAL_WHEEL",
        "earnings_note": f"CLEAR: ปลอดวันประกาศงบตลอดช่วง {dte_days} DTE",
    }


def fetch_ticker_earnings_blackout(
    symbol: str,
    trade_date: str,
    dte_days: int = 35,
    post_earnings_cooldown_days: int = 3,
) -> dict:
    sym_upper = symbol.upper().strip()
    if sym_upper in ETF_EXEMPT_SYMBOLS:
        return evaluate_earnings_blackout_from_dates(sym_upper, trade_date, [], dte_days)

    try:
        extracted_dates = wheel_data.earnings_dates(sym_upper)
    except VendorError:
        extracted_dates = []  # reported as EARNINGS_DATES_UNAVAILABLE, never as CLEAR

    return evaluate_earnings_blackout_from_dates(
        symbol=sym_upper,
        trade_date=trade_date,
        earnings_dates=extracted_dates,
        dte_days=dte_days,
        post_earnings_cooldown_days=post_earnings_cooldown_days,
    )


def evaluate_option_roll_mechanics(
    option_type: Literal["CALL", "PUT"],
    spot: float,
    current_strike: float,
    remaining_dte: int,
    entry_premium: float,
    iv_annual: float,
    roll_mode: RollModeType = "AUTO_CDC",
    cdc_zone: Literal["GREEN", "YELLOW", "BLUE", "RED"] = "YELLOW",
    days_to_next_earnings: int | None = None,
    roll_add_days: int = 30,
    strike_step: float = 5.0,
    min_net_credit: float = 0.00,
    max_scan_steps: int = 15,
) -> dict:
    new_dte = remaining_dte + roll_add_days
    effective_mode = (
        resolve_optimal_roll_mode(option_type, cdc_zone)
        if roll_mode == "AUTO_CDC"
        else roll_mode
    )

    curr_iv = estimate_skewed_iv(iv_annual, spot, current_strike, option_type)
    if option_type == "CALL":
        curr = compute_black_scholes_call(spot, current_strike, remaining_dte, curr_iv)
        curr_mark = curr["call_premium_per_share"]
        curr_delta = curr["call_delta"]
    else:
        curr = compute_black_scholes_put(spot, current_strike, remaining_dte, curr_iv)
        curr_mark = curr["put_premium_per_share"]
        curr_delta = curr["put_delta"]

    unrealized_pnl = entry_premium - curr_mark
    profit_capture_pct = (unrealized_pnl / entry_premium) * 100.0 if entry_premium > 0 else 0.0

    is_profit_50 = profit_capture_pct >= 50.0
    is_21dte = remaining_dte <= 21
    is_challenged_call = option_type == "CALL" and (spot >= current_strike * 0.985 or abs(curr_delta) >= 0.45)
    is_challenged_put = option_type == "PUT" and (spot <= current_strike * 1.015 or abs(curr_delta) >= 0.45)
    is_earnings_imminent = (
        days_to_next_earnings is not None and 0 <= days_to_next_earnings <= remaining_dte
    )

    valid_credit_candidates = []
    all_scanned_candidates = []

    for step_idx in range(1, max_scan_steps + 1):
        if option_type == "CALL":
            cand_strike = current_strike + (step_idx * strike_step)
            cand_iv = estimate_skewed_iv(iv_annual, spot, cand_strike, "CALL")
            nxt = compute_black_scholes_call(spot, cand_strike, new_dte, cand_iv)
            sto_prem = nxt["call_premium_per_share"]
            new_delta = nxt["call_delta"]
            net_credit = round(sto_prem - curr_mark, 2)
            cand_info = {
                "step_index": step_idx,
                "new_strike": round(cand_strike, 2),
                "strike_distance_moved": round(cand_strike - current_strike, 2),
                "new_dte": new_dte,
                "skew_adjusted_iv_pct": round(cand_iv * 100.0, 2),
                "btc_cost": round(curr_mark, 2),
                "sto_premium": sto_prem,
                "net_credit_per_share": net_credit,
                "new_delta": new_delta,
                "new_effective_exit": round(cand_strike + entry_premium + net_credit, 2),
                "total_benefit_per_share": round((cand_strike - current_strike) + net_credit, 2),
            }
        else:
            cand_strike = current_strike - (step_idx * strike_step)
            if cand_strike <= strike_step:
                break
            cand_iv = estimate_skewed_iv(iv_annual, spot, cand_strike, "PUT")
            nxt = compute_black_scholes_put(spot, cand_strike, new_dte, cand_iv)
            sto_prem = nxt["put_premium_per_share"]
            new_delta = nxt["put_delta"]
            net_credit = round(sto_prem - curr_mark, 2)
            cand_info = {
                "step_index": step_idx,
                "new_strike": round(cand_strike, 2),
                "strike_distance_moved": round(current_strike - cand_strike, 2),
                "new_dte": new_dte,
                "skew_adjusted_iv_pct": round(cand_iv * 100.0, 2),
                "btc_cost": round(curr_mark, 2),
                "sto_premium": sto_prem,
                "net_credit_per_share": net_credit,
                "new_delta": new_delta,
                "new_net_breakeven": round(cand_strike - (entry_premium + net_credit), 2),
                "total_benefit_per_share": round((current_strike - cand_strike) + net_credit, 2),
            }

        all_scanned_candidates.append(cand_info)
        if net_credit >= min_net_credit:
            valid_credit_candidates.append(cand_info)

    if valid_credit_candidates:
        mode_map = {
            "MAX_NET_CREDIT": valid_credit_candidates[0],
            "BALANCED_50_50": valid_credit_candidates[len(valid_credit_candidates) // 2],
            "MAX_STRIKE_SHIFT": valid_credit_candidates[-1],
        }
        chosen_roll = dict(mode_map[effective_mode])
        chosen_roll["selected_roll_mode"] = effective_mode
    else:
        fallback = dict(all_scanned_candidates[0])
        fallback["selected_roll_mode"] = "FALLBACK_MIN_DEBIT"
        mode_map = {
            "MAX_NET_CREDIT": fallback,
            "BALANCED_50_50": fallback,
            "MAX_STRIKE_SHIFT": fallback,
        }
        chosen_roll = fallback

    if is_earnings_imminent:
        verdict = (
            f"CLOSE_BEFORE_EARNINGS (Earnings in {days_to_next_earnings}d <= {remaining_dte} DTE "
            f"| BTC @ ${curr_mark:.2f} to avoid Binary Gap Risk)"
        )
    elif is_profit_50 and not (is_challenged_call or is_challenged_put):
        verdict = f"TAKE_PROFIT_50_PCT (BTC @ ${curr_mark:.2f} | +{profit_capture_pct:.1f}%)"
    elif is_challenged_call:
        verdict = (
            f"ROLL_UP [{effective_mode}] -> ${chosen_roll['new_strike']:.0f} CALL "
            f"(Net Credit ${chosen_roll['net_credit_per_share']:+.2f})"
        )
    elif is_challenged_put:
        verdict = (
            f"ROLL_DOWN [{effective_mode}] -> ${chosen_roll['new_strike']:.0f} PUT "
            f"(Net Credit ${chosen_roll['net_credit_per_share']:+.2f})"
        )
    elif is_21dte:
        verdict = f"TIME_GATE_21DTE_ROLL [{effective_mode}] -> ${chosen_roll['new_strike']:.0f}"
    else:
        verdict = "HOLD_EXISTING_CONTRACT"

    return {
        "option_type": option_type,
        "spot_price": round(spot, 2),
        "current_strike": current_strike,
        "remaining_dte": remaining_dte,
        "cdc_zone": cdc_zone,
        "requested_roll_mode": roll_mode,
        "effective_roll_mode": effective_mode,
        "current_mark_price": round(curr_mark, 2),
        "current_delta": curr_delta,
        "profit_capture_pct": round(profit_capture_pct, 1),
        "decision_verdict": verdict,
        "selected_roll_plan": chosen_roll,
        "mode_comparison": mode_map,
    }


def analyze_thai_dividend_and_reit_regime(
    symbol: str,
    trade_date: str,
    fair_value_estimate: float | None = None,
    thai_10y_bond_yield_pct: float = 2.45,
) -> dict:
    sym_upper = symbol.upper().strip()
    if not sym_upper.endswith(".BK"):
        sym_upper = f"{sym_upper}.BK"

    is_reit = sym_upper in THAI_REIT_AND_IFF_SYMBOLS
    asset_class = "THAI_REIT_IFF" if is_reit else "THAI_DIVIDEND_EQUITY"

    safe_date = _day(trade_date)

    try:
        close = wheel_data.weekly_closes(sym_upper, safe_date)
    except VendorError as exc:
        return {"symbol": sym_upper, "error": f"Price history unavailable: {exc}"}
    if len(close) < 60:
        return {"symbol": sym_upper, "error": "Insufficient weekly history for Thai symbol"}

    trend = _weekly_trend(close)
    spot, ema100, ema200 = trend["spot"], trend["ema100"], trend["ema200"]
    peak_2yr, dd_pct = trend["peak_2yr"], trend["dd_pct"]
    dist_ema100_pct, cdc_zone = trend["dist_ema100_pct"], trend["cdc_zone"]

    ttm_div_dps = 0.0
    last_xd_date = None
    dividend_note = None
    try:
        divs = wheel_data.dividends(sym_upper)
    except VendorError as exc:
        divs = pd.Series(dtype=float)
        dividend_note = f"Dividend history unavailable: {exc}"
    if not divs.empty:
        cutoff_end = pd.to_datetime(safe_date)
        cutoff_start = cutoff_end - pd.DateOffset(days=370)
        ttm_div_dps = float(divs[(divs.index >= cutoff_start) & (divs.index <= cutoff_end)].sum())
        past_divs = divs[divs.index <= cutoff_end]
        if not past_divs.empty:
            last_xd_date = past_divs.index[-1].strftime("%Y-%m-%d")

    gross_yield_pct = (ttm_div_dps / spot) * 100.0 if spot > 0 else 0.0
    net_yield_after_tax_pct = gross_yield_pct * 0.90
    yield_spread_vs_bond = net_yield_after_tax_pct - thai_10y_bond_yield_pct

    anchor_fair_price = fair_value_estimate if fair_value_estimate else ((ema100 * 0.65) + (peak_2yr * 0.35))
    mos_10pct_target_entry = round(anchor_fair_price * 0.90, 2)
    current_mos_pct = round(((anchor_fair_price - spot) / anchor_fair_price) * 100.0, 2)

    min_attractive_yield = 6.5 if is_reit else 4.2
    near_ema200 = ema200 is not None and spot <= ema200 * 1.01
    if (dd_pct <= -30.0 or near_ema200) and gross_yield_pct >= min_attractive_yield:
        tier = "TIER_3_DEEP_VALUE"
        action = f"BUY 100% SPOT (Deep Value & Gross Yield {gross_yield_pct:.2f}%)"
    elif (spot <= ema100 or current_mos_pct >= 10.0) and gross_yield_pct >= (min_attractive_yield - 0.8):
        tier = "TIER_2_ACCUMULATION"
        action = f"ACCUMULATE 50% (At EMA100 / 10% MoS | Net Yield {net_yield_after_tax_pct:.2f}%)"
    elif spot <= ema100 * 1.04:
        tier = "TIER_1_WATCH_ZONE"
        action = f"SET LIMIT @ ฿{min(ema100, mos_10pct_target_entry):.2f} (Wait for EMA100 / 10% MoS)"
    else:
        tier = "TIER_0_EXTENDED"
        action = f"HOLD FOR DIVIDEND / TRIM @ +15% GAP (Wait pullback to ฿{ema100:.2f})"

    return {
        "Ticker": sym_upper,
        "Market": asset_class,
        "Currency": "THB (฿)",
        "Spot": round(spot, 2),
        "Weekly EMA100": round(ema100, 2),
        "Weekly EMA200": round(ema200, 2) if ema200 is not None else None,
        "Dist EMA100 (%)": round(dist_ema100_pct, 2),
        "Drawdown 2Y (%)": round(dd_pct, 2),
        "CDC Zone": cdc_zone,
        "TTM Div (฿/sh)": round(ttm_div_dps, 2),
        "Gross Yield (%)": round(gross_yield_pct, 2),
        "Net Yield 90% (%)": round(net_yield_after_tax_pct, 2),
        "Yield Spread vs 10Y Bond (%)": round(yield_spread_vs_bond, 2),
        "10% MoS Entry (฿)": mos_10pct_target_entry,
        "Current MoS (%)": current_mos_pct,
        "Last XD Date": last_xd_date,
        "Tactical Tier": tier,
        "Execution Action": action,
        **({"Note": dividend_note} if dividend_note else {}),
    }


CONTRACT_SHARES = 100
IV_SOURCE = "realised 30-day volatility x 1.15, floor 22% (an estimate, not a market quote)"


def _tier(spot: float, ema100: float, ema200: float | None, dd_pct: float) -> tuple[str, str]:
    """The tactical tier and the rule that set it, checked deepest first.

    A drawdown rule can set a deeper tier than the EMA distance alone would: a
    stock 5% above its EMA100 but 26% off its 2-year high is TIER_2.
    """
    dist = (spot / ema100 - 1.0) * 100.0
    if dd_pct <= -35.0:
        return "TIER_3_DEEP_VALUE", f"2-year drawdown {dd_pct:.1f}% <= -35%"
    if ema200 is not None and spot <= ema200 * 1.01:
        return "TIER_3_DEEP_VALUE", f"close within 1% of or below the Weekly EMA200 ({ema200:.2f})"
    if spot <= ema100:
        return "TIER_2_ACCUMULATION", f"close at or below the Weekly EMA100 ({dist:+.1f}%)"
    if dd_pct <= -20.0:
        return "TIER_2_ACCUMULATION", f"2-year drawdown {dd_pct:.1f}% <= -20% (overrides EMA100 distance {dist:+.1f}%)"
    if dist <= 5.0:
        return "TIER_1_WATCH_ZONE", f"close {dist:+.1f}% above the Weekly EMA100 (0% to 5%)"
    return "TIER_0_EXTENDED", f"close {dist:+.1f}% above the Weekly EMA100 (> 5%)"


def _open_option_reviews(options: list[dict], spot: float, est_iv: float, cdc_zone: str,
                         trade_date: str, days_to_earnings: int | None) -> list[dict]:
    """The roll / take-profit / close verdict for each open short option."""
    reviews = []
    ref = pd.Timestamp(trade_date)
    for o in options:
        remaining = int((pd.Timestamp(o["expiry"]) - ref).days)
        base = {"option_type": o["option_type"], "strike": o["strike"], "expiry": o["expiry"],
                "contracts": o["contracts"], "remaining_dte": remaining}
        if remaining <= 0:
            reviews.append({**base, "decision_verdict": "EXPIRED_OR_EXPIRING_TODAY (settle or record assignment)"})
            continue
        res = evaluate_option_roll_mechanics(
            option_type=o["option_type"], spot=spot, current_strike=float(o["strike"]),
            remaining_dte=remaining, entry_premium=float(o["entry_premium"]), iv_annual=est_iv,
            roll_mode="AUTO_CDC", cdc_zone=cdc_zone, days_to_next_earnings=days_to_earnings,
            strike_step=5.0 if spot >= 100 else 2.5,
        )
        reviews.append({**base,
                        "current_mark_price": res["current_mark_price"],
                        "profit_capture_pct": res["profit_capture_pct"],
                        "effective_roll_mode": res["effective_roll_mode"],
                        "decision_verdict": res["decision_verdict"],
                        "selected_roll_plan": res["selected_roll_plan"]})
    return reviews


def analyze_weekly_wheel_regime(
    symbol: str,
    trade_date: str,
    dte_days: int = 35,
    shares_held: float | None = None,
    open_options: list[dict] | None = None,
    cash_available: float | None = None,
) -> dict:
    """Weekly EMA100/200 tier, CDC zone, earnings gate and the wheel leg for one symbol.

    ``shares_held`` is None when holdings are unknown, so a covered call is then
    marked as needing 100 shares rather than recommended outright. ``open_options``
    are the caller's open short options on this symbol, each reviewed for a roll;
    shares already under an open call are not offered again. ``cash_available`` is
    free cash after the collateral of every open short put (None when unknown); a
    new cash-secured put must fit in it.
    Prices are as traded and cut at ``trade_date``. A symbol whose data cannot be
    read comes back with an ``error`` key instead of a recommendation.
    """
    sym = symbol.upper().strip()
    day = _day(trade_date)
    try:
        close = wheel_data.weekly_closes(sym, day)
        daily = wheel_data.daily_closes(sym, day)
    except VendorError as exc:
        return {"symbol": sym, "error": f"Price history unavailable: {exc}"}
    if len(close) < 100:
        return {"symbol": sym, "error": "Insufficient weekly history (needs 100 weeks)"}

    trend = _weekly_trend(close)
    spot, ema100, ema200, dd_pct = trend["spot"], trend["ema100"], trend["ema200"], trend["dd_pct"]
    cdc = trend["cdc_zone"]

    log_rets = np.log(daily / daily.shift(1)).dropna().tail(30)
    est_iv = max(float(log_rets.std() * np.sqrt(252)) * 1.15, 0.22)
    step = 5.0 if spot >= 100 else 2.5

    open_options = open_options or []
    calls_open = sum(int(o["contracts"]) for o in open_options if o["option_type"] == "CALL")
    puts_open = sum(int(o["contracts"]) for o in open_options if o["option_type"] == "PUT")
    free_shares = None if shares_held is None else max(shares_held - calls_open * CONTRACT_SHARES, 0.0)

    tier, tier_reason = _tier(spot, ema100, ema200, dd_pct)
    gate = fetch_ticker_earnings_blackout(sym, trade_date=day, dte_days=dte_days)
    blocked = gate["is_blackout"]
    notes = []
    opt = None
    if puts_open:
        notes.append(f"{puts_open} short put(s) already open on {sym}: a new CSP adds to that exposure")

    def csp_unaffordable(put: dict) -> str | None:
        """Why a new CSP cannot be secured by the free cash, or None when it can or cash is unknown."""
        need = put["cash_collateral_1_contract"]
        if cash_available is not None and cash_available < need:
            return f"CSP needs {need:,.0f} cash per contract, {cash_available:,.0f} free"
        return None

    if tier == "TIER_3_DEEP_VALUE":
        action, vehicle = "BUY 100% SPOT", "EQUITY_SPOT"
    elif tier == "TIER_2_ACCUMULATION":
        # In a weekly downtrend (CDC RED) the put is struck deeper, at 85% of spot.
        put_mult = 0.85 if cdc == "RED" else 0.90
        opt = compute_black_scholes_put(spot, math.floor((spot * put_mult) / step) * step, dte_days, est_iv)
        if cdc == "RED":
            notes.append("CDC RED: put struck at 85% of spot instead of 90%")
        short_cash = csp_unaffordable(opt)
        if blocked:
            action, vehicle = "BUY 50% SPOT ONLY (CSP blocked by earnings gate)", "EQUITY_SPOT"
        elif short_cash:
            action, vehicle = f"BUY 50% SPOT ONLY ({short_cash})", "EQUITY_SPOT"
        else:
            action, vehicle = "BUY 50% SPOT + SELL CSP", "HYBRID_SPOT_PLUS_CSP"
    elif tier == "TIER_1_WATCH_ZONE":
        if cdc == "RED":
            # A falling weekly trend: no new put at the EMA100 until it turns.
            action, vehicle = "WAIT (CDC RED: no new CSP until the weekly trend turns)", "NONE"
        else:
            opt = compute_black_scholes_put(spot, math.floor(ema100 / step) * step, dte_days, est_iv)
            short_cash = csp_unaffordable(opt)
            if blocked:
                action, vehicle = "WAIT (CSP blocked by earnings gate)", "NONE"
            elif short_cash:
                action, vehicle = f"WAIT ({short_cash})", "NONE"
            else:
                action, vehicle = "SELL CSP @ WEEKLY EMA100", "CASH_SECURED_PUT"
    else:  # TIER_0_EXTENDED
        # A strong weekly uptrend (CDC GREEN) gets a further strike so the call
        # caps less of the move; otherwise 0.20, or 0.24 when far extended.
        if cdc == "GREEN":
            target_delta = 0.15
            notes.append("CDC GREEN: call targets delta 0.15 to leave room for the uptrend")
        else:
            target_delta = 0.24 if trend["dist_ema100_pct"] >= 15.0 else 0.20
        opt = select_optimal_covered_call_strike(spot, dte_days, est_iv, target_delta=target_delta, strike_step=step)
        if opt["delta_off_target"]:
            notes.append(f"call delta {opt['call_delta']:+.2f} is off the {target_delta:.2f} target: "
                         "no strike within range reached it (high volatility)")
        if free_shares is not None and free_shares < CONTRACT_SHARES:
            covered = f", {calls_open * CONTRACT_SHARES} already under open calls" if calls_open else ""
            action, vehicle = (f"NO NEW COVERED CALL (holds {shares_held:g} shares{covered}; "
                               f"needs {CONTRACT_SHARES} uncovered)"), "NONE"
        elif blocked:
            action, vehicle = "HOLD SHARES (covered call blocked by earnings gate)", "NONE"
        elif free_shares is None:
            action, vehicle = "SELL COVERED CALL (only if 100+ shares are held)", "COVERED_CALL_OVERWRITE"
            notes.append("holdings unknown: a call without 100 shares per contract is a naked call")
        else:
            contracts = int(free_shares // CONTRACT_SHARES)
            action, vehicle = f"SELL {contracts} COVERED CALL(S)", "COVERED_CALL_OVERWRITE"

    if opt is not None and gate.get("days_to_next_earnings") is not None and gate["days_to_next_earnings"] <= dte_days:
        notes.append("earnings fall inside the option's life: real IV is likely higher than the estimate")

    result = {
        "symbol": sym,
        "close": round(spot, 2),
        "weekly_ema12": round(trend["ema12"], 2),
        "weekly_ema26": round(trend["ema26"], 2),
        "weekly_ema100": round(ema100, 2),
        "dist_to_ema100_pct": round(trend["dist_ema100_pct"], 2),
        "weekly_ema200": round(ema200, 2) if ema200 is not None else None,
        "max_2yr_drawdown_pct": round(dd_pct, 2),
        "cdc_zone": cdc,
        "tactical_regime_tier": tier,
        "tier_reason": tier_reason,
        "earnings_blackout_gate": gate,
        "wheel_action": action,
        "execution_vehicle": vehicle,
        "shares_held": shares_held,
        "shares_free_for_calls": free_shares,
        "cash_available": cash_available,
        "wheel_option_recommendation": opt,
        "estimated_iv_pct": round(est_iv * 100.0, 2),
        "iv_source": IV_SOURCE,
        "notes": notes,
    }
    if open_options:
        result["open_option_reviews"] = _open_option_reviews(
            open_options, spot, est_iv, cdc, day, gate.get("days_to_next_earnings"))
    return result


def free_cash(portfolio) -> float | None:
    """Cash left after the collateral of every open short put; None when cash is not given."""
    if portfolio.cash is None:
        return None
    held = sum(o.strike * CONTRACT_SHARES * o.contracts for o in portfolio.options if o.option_type == "PUT")
    return float(portfolio.cash - held)


def _book_for(book: dict | None, symbol: str) -> tuple[float | None, list[dict], float | None]:
    """Shares held, open options on ``symbol`` and free cash from a portfolio dump.

    No book gives (None, [], None): holdings and cash unknown, not zero.
    """
    if not book:
        return None, [], None
    from tradingagents.portfolio import PortfolioContext
    portfolio = PortfolioContext.model_validate(book)
    return (portfolio.shares_in(symbol), [o.model_dump() for o in portfolio.options_in(symbol)],
            free_cash(portfolio))


@tool
def get_weekly_macro_drawdown_regime(
    symbol: Annotated[str, InjectedState("company_of_interest")],
    benchmark_ticker: Annotated[str, "Macro benchmark to read alongside the asset, e.g. QQQ"] = "QQQ",
    dte_days: Annotated[int, "Target days to expiry for the option leg"] = 35,
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
    portfolio_book: Annotated[dict, InjectedState("portfolio_book")] = None,
) -> str:
    """
    Weekly EMA100/EMA200 tier (with the rule that set it), 2-year drawdown, Weekly
    CDC Action Zone, earnings blackout gate and the options-wheel action for the
    instrument under analysis and a macro benchmark, as of the analysis date.
    With a portfolio it checks the 100 shares a covered call needs and reviews
    each open short option for a roll, take-profit or close. Option premiums are
    Black-Scholes estimates from realised volatility, not market quotes. Thai
    (.BK) listings have no single-stock options, so for them it reports the
    dividend/REIT regime.
    """
    target = symbol.upper().strip()
    if target.endswith(".BK"):
        return (
            f"As of {_day(trade_date)}:\n"
            f"- Target Asset ({target}, SET: no listed single-stock options; dividend regime instead): "
            f"{analyze_thai_dividend_and_reit_regime(target, trade_date)}"
        )
    shares, options, cash = _book_for(portfolio_book, target)
    benchmark = analyze_weekly_wheel_regime(benchmark_ticker, trade_date, dte_days)
    asset = analyze_weekly_wheel_regime(target, trade_date, dte_days, shares_held=shares,
                                        open_options=options, cash_available=cash)
    return (
        f"As of {_day(trade_date)}:\n"
        f"- Macro Benchmark ({benchmark_ticker}, for context only): {benchmark}\n"
        f"- Target Asset ({target}): {asset}"
    )
