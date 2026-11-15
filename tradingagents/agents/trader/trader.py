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
    "Options-wheel mode is on. Read the Weekly EMA tier, earnings blackout gate and wheel leg "
    "from the technical market report and apply these rules, highest priority first:\n"
    "- EARNINGS BLACKOUT: if the gate is blocked (PRE_EARNINGS_BLACKOUT, POST_EARNINGS_COOLDOWN "
    "or EARNINGS_DATES_UNAVAILABLE), do not open new short calls or puts; hold option collateral "
    "in cash or T-bills until the safe resume date, or until the date is confirmed.\n"
    "- TIER_0_EXTENDED (more than 5% above the Weekly EMA100): covered call, delta about +0.20 to "
    "+0.25, about 35 DTE.\n"
    "- TIER_1_WATCH_ZONE (0% to 5% above the Weekly EMA100): cash-secured put struck near the "
    "Weekly EMA100, about 35 DTE.\n"
    "- TIER_2_ACCUMULATION (at or below the Weekly EMA100, or a 20% drawdown): half spot, half "
    "out-of-the-money cash-secured put.\n"
    "- TIER_3_DEEP_VALUE (at or below the Weekly EMA200, or a 35% drawdown): spot purchase.\n"
    "- Thai (.BK) listings have no single-stock options: use the dividend regime's tier and "
    "entry levels for spot only.\n"
    "Option premiums in the report are model estimates; size from them, but say the live quote "
    "must be checked. Fill macro_regime_tier, execution_vehicle and earnings_blackout_status. "
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
