#!/usr/bin/env python3
"""
ไฟล์: wheel_full_cycle_scanner.py
วิธีรัน:
  python wheel_full_cycle_scanner.py --date 2026-11-13 --output-dir reports [--notify-telegram]

สแกน Weekly EMA100/200 + The Wheel ของหุ้นสหรัฐฯ ไม่ต้องใช้ API key ของ AI
ราคาออปชันเป็นค่าประมาณจาก Black-Scholes (realised volatility) ไม่ใช่ราคาตลาดจริง
--notify-telegram ต้องตั้ง TELEGRAM_BOT_TOKEN และ TELEGRAM_CHAT_ID ใน environment
"""

import argparse
from datetime import datetime, timezone
import html
import os
from pathlib import Path
import urllib.parse
import urllib.request

import pandas as pd

from tradingagents.agents.quant_wheel_tools import analyze_weekly_wheel_regime

US_WATCHLIST = ["QQQ", "NVDA", "GOOGL", "AMD", "ASML", "MSFT", "AMZN", "META", "AAPL"]


def _wheel_row(res: dict) -> dict:
    """One scanner row from analyze_weekly_wheel_regime's result."""
    tier = res["tactical_regime_tier"]
    gate = res["earnings_blackout_gate"]
    opt = res["wheel_option_recommendation"]
    row = {
        "Ticker": res["symbol"],
        "Spot ($)": res["close"],
        "Weekly EMA100 ($)": res["weekly_ema100"],
        "Dist EMA100 (%)": res["dist_to_ema100_pct"],
        "Drawdown 2Y (%)": res["max_2yr_drawdown_pct"],
        "CDC Zone": res["cdc_zone"],
        "Tactical Tier": tier,
        "Earnings Gate": gate["blackout_status"],
    }
    if tier == "TIER_0_EXTENDED":
        action = "SELL COVERED CALL" if not gate["is_blackout"] else f"HOLD ({gate['blackout_status']})"
        row.update({
            "Wheel Action": action,
            "Strike": f"${opt['call_strike']:.0f} CALL",
            "Delta": f"+{opt['call_delta']:.2f}",
            "Premium ($/sh)": opt["call_premium_per_share"],
            "Target Exit / Entry ($)": opt["effective_call_away_price"],
            "Annualized Yield (%)": opt["annualized_yield_pct"],
        })
    else:
        if gate["is_blackout"]:
            resume = gate["safe_resume_date"] or "earnings date is confirmed"
            action = f"SPOT ONLY (NO NEW CSP UNTIL {resume})"
        else:
            action = {
                "TIER_1_WATCH_ZONE": "SELL CSP @ EMA100",
                "TIER_2_ACCUMULATION": "50% SPOT + 50% CSP",
            }.get(tier, "100% BUY SPOT")
        row.update({
            "Wheel Action": action,
            "Strike": f"${opt['strike_price']:.0f} PUT",
            "Delta": f"{opt['put_delta']:.2f}",
            "Premium ($/sh)": opt["put_premium_per_share"],
            "Target Exit / Entry ($)": opt["net_breakeven_price"],
            "Annualized Yield (%)": opt["annualized_yield_pct"],
        })
    return row


def run_full_wheel_portfolio_scan(
    tickers: list[str],
    trade_date: str,
    dte_days: int = 35,
) -> pd.DataFrame:
    rows, skipped = [], []
    for sym in tickers:
        res = analyze_weekly_wheel_regime(sym, trade_date, dte_days)
        if "error" in res:
            skipped.append(f"{sym}: {res['error']}")
            continue
        rows.append(_wheel_row(res))
    for line in skipped:
        print(f"[!] ข้าม {line}")
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    """A GitHub-flavoured Markdown table, without the optional tabulate dependency."""
    if df.empty:
        return "_ไม่มีข้อมูล_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def telegram_summary(title: str, df: pd.DataFrame, columns: list[str]) -> str:
    """A short HTML message: one line per row with the given columns."""
    lines = [f"<b>{html.escape(title)}</b>"]
    for _, r in df.iterrows():
        lines.append(" | ".join(html.escape(str(r[c])) for c in columns if c in df.columns))
    lines.append("<i>Research only, not investment advice. Option prices are model estimates.</i>")
    return "\n".join(lines)


def send_telegram_message(text: str) -> bool:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        print("[!] ไม่พบ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID (ข้ามการส่ง Telegram)")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text[:4000],  # Telegram's message limit is 4096 characters
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except OSError as exc:  # the token is in the URL: never print the request itself
        print(f"[!] ส่ง Telegram ไม่สำเร็จ: {type(exc).__name__}")
        return False


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-Cycle Wheel & Weekly EMA100/200 Scanner")
    parser.add_argument("--date", type=str, default=today_utc())
    parser.add_argument("--dte", type=int, default=35)
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--notify-telegram", action="store_true")
    args = parser.parse_args()

    df_wheel = run_full_wheel_portfolio_scan(US_WATCHLIST, trade_date=args.date, dte_days=args.dte)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_wheel.to_csv(out_dir / f"wheel_scan_{args.date}.csv", index=False, encoding="utf-8-sig")
    df_wheel.to_json(out_dir / "latest_wheel_scan.json", orient="records", indent=2, force_ascii=False)
    md_content = f"# Weekly EMA100/200 & The Wheel Scan ({args.date})\n\n{markdown_table(df_wheel)}\n"
    (out_dir / f"wheel_scan_{args.date}.md").write_text(md_content, encoding="utf-8")
    (out_dir / "latest_wheel_scan.md").write_text(md_content, encoding="utf-8")

    print(df_wheel.to_string(index=False))
    if args.notify_telegram:
        send_telegram_message(telegram_summary(
            f"Wheel scan {args.date}", df_wheel, ["Ticker", "Tactical Tier", "Earnings Gate", "Wheel Action"]))
