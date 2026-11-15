#!/usr/bin/env python3
"""
ไฟล์: dual_market_wheel_dividend_scanner.py
วิธีรัน:
  python dual_market_wheel_dividend_scanner.py --date 2026-11-13 --output-dir reports [--notify-telegram]

สแกน 2 ตลาด: US Options Wheel + หุ้นปันผล/REIT ไทย (.BK) ไม่ต้องใช้ API key ของ AI
มูลค่าเหมาะสม (fair value) ใน THAI_FAIR_VALUE_ANCHORS เป็นค่าที่ผู้ใช้กำหนดเอง ควรทบทวนก่อนใช้
"""

import argparse
from pathlib import Path

import pandas as pd

from wheel_full_cycle_scanner import (
    US_WATCHLIST,
    markdown_table,
    run_full_wheel_portfolio_scan,
    send_telegram_message,
    telegram_summary,
    today_utc,
)
from tradingagents.agents.quant_wheel_tools import analyze_thai_dividend_and_reit_regime

THAI_FAIR_VALUE_ANCHORS = {
    "PTTEP.BK": 148.00,
    "ADVANC.BK": 285.00,
    "KTC.BK": 48.00,
    "TOG.BK": 11.20,
    "SMPC.BK": 10.80,
    "JMT.BK": 18.50,
    "FTREIT.BK": 11.00,
    "WHAIR.BK": 7.40,
    "DIF.BK": 9.80,
}


def run_thai_dividend_and_reit_scan(
    thai_tickers: list[str],
    trade_date: str,
) -> pd.DataFrame:
    rows = []
    for sym in thai_tickers:
        fv = THAI_FAIR_VALUE_ANCHORS.get(sym.upper())
        res = analyze_thai_dividend_and_reit_regime(
            symbol=sym,
            trade_date=trade_date,
            fair_value_estimate=fv,
        )
        if "error" in res:
            print(f"[!] ข้าม {sym}: {res['error']}")
            continue
        rows.append(res)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Market US Options Wheel + Thai Dividend/REIT Scanner")
    parser.add_argument("--date", type=str, default=today_utc())
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--notify-telegram", action="store_true")
    args = parser.parse_args()

    df_us = run_full_wheel_portfolio_scan(US_WATCHLIST, trade_date=args.date, dte_days=35)
    df_th = run_thai_dividend_and_reit_scan(list(THAI_FAIR_VALUE_ANCHORS), trade_date=args.date)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_th.to_csv(out_dir / f"thai_div_reit_scan_{args.date}.csv", index=False, encoding="utf-8-sig")
    df_th.to_json(out_dir / "latest_thai_dividend_scan.json", orient="records", indent=2, force_ascii=False)
    md = f"# Thai Dividend & REIT Scan ({args.date})\n\n{markdown_table(df_th)}\n"
    (out_dir / "latest_thai_dividend_scan.md").write_text(md, encoding="utf-8")

    print("\n=== [1] US OPTIONS WHEEL SCAN ===")
    print(df_us.to_string(index=False))
    print("\n=== [2] THAI DIVIDEND & REIT SCAN (WEEKLY EMA100/200 + 10% MoS) ===")
    print(df_th.to_string(index=False))
    if args.notify_telegram:
        send_telegram_message(telegram_summary(
            f"Thai dividend/REIT scan {args.date}", df_th, ["Ticker", "Tactical Tier", "Net Yield 90% (%)", "Execution Action"]))
