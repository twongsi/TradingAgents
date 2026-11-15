"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.config import get_config

# Added to the Trader's instructions when the run's ``wheel_strategy`` is on. It
# reads the wheel tool's output from the technical market report.
WHEEL_MANDATE = (
    "Options-wheel mode is on. The technical market report carries the wheel tool's output for the "
    "target asset: its tier and the rule that set it (tier_reason), CDC zone, earnings gate, "
    "wheel_action, shares held and, when given, a review of each open short option. Follow "
    "wheel_action; it already applies these rules, highest priority first:\n"
    "- EARNINGS GATE: when blocked (PRE_EARNINGS_BLACKOUT, POST_EARNINGS_COOLDOWN or "
    "EARNINGS_DATES_UNAVAILABLE) open no new short call or put.\n"
    "- COVERED CALLS need 100 shares per contract. Never propose a call on fewer shares; with "
    "holdings unknown, state that 100 shares are required.\n"
    "- TIER_3_DEEP_VALUE (35% drawdown, or at the Weekly EMA200): buy spot only, no option.\n"
    "- TIER_2_ACCUMULATION (at or below the Weekly EMA100, or a 20% drawdown): half spot plus a "
    "cash-secured put at 90% of spot (85% when CDC is RED).\n"
    "- TIER_1_WATCH_ZONE (0% to 5% above the Weekly EMA100): cash-secured put near the EMA100; "
    "when CDC is RED, wait instead.\n"
    "- TIER_0_EXTENDED (more than 5% above the Weekly EMA100): covered call, delta 0.20 (0.24 when "
    "15% extended, 0.15 when CDC is GREEN).\n"
    "- OPEN OPTIONS: act on each review's verdict (CLOSE_BEFORE_EARNINGS, TAKE_PROFIT_50_PCT, "
    "ROLL_UP / ROLL_DOWN, TIME_GATE_21DTE_ROLL, HOLD_EXISTING_CONTRACT).\n"
    "- Thai (.BK) listings have no single-stock options: spot only, from the dividend regime.\n"
    "Premiums are model estimates from realised volatility; quote them as estimates and say the "
    "live option price must be checked. Fill macro_regime_tier, execution_vehicle and "
    "earnings_blackout_status. "
)


def wheel_mandate() -> str:
    """The wheel rules when the run's wheel_strategy is on, else nothing."""
    return WHEEL_MANDATE if get_config().get("wheel_strategy") else ""


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    + grounding
                    # Entry/stop are numeric price fields. Asking for concrete
                    # levels invites a percentage ("15%"), which is not a price
                    # and fails the structured parse (#1288).
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; convert a percentage distance to the price level it "
                    "implies, or omit the field if you cannot state a number. "
                    + wheel_mandate()
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    "Make an informed, strategic trading decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, "
                    "sized by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, against the plan and the price structure\n"
                    "- **Entry Price**, **Stop Loss**, **Position Sizing**: when you can state them"
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
        }

    return trader_node
