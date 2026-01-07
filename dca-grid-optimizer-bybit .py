from __future__ import annotations

"""
Python translation of the provided Pine Script strategy:
Grid Strategy with MA (SHORT ONLY, Single Entry) - 5m hardcoded + 1% TP

Key fidelity points:
- Strictly 5m candles; aborts if another interval is requested.
- Short-only, single entry (no pyramiding).
- Position size: 100% of current equity notional per entry (percent_of_equity=100).
- Commission: percent, default 0.01% (Pine used commission_value=0.01 with commission.percent).
- Slippage: configurable in bps (Pine slippage=1 tick; default 0 here).
- MA length 100 using RMA (Wilder), premium/discount grids use EMA(5) of RMA-adjusted levels.
- Entry on crossover of high above premium zones (levels 1..8, priority 8->1).
- Take-profit: fixed 1% limit from avg entry (short_tp_price = entry * (1 - 0.01)).
- Fallback exit on crossunder of low below any discount zone (1..8) or main RMA (coverCondition9).
- Date filter defaults: start 2024-01-01, end 2027-01-01, enabled by default.
- No pyramiding, no additional adds, no optimizer—single deterministic backtest run.
"""

import argparse
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests

# =============================
# Rate Limiter
# =============================


class RateLimiter:
    """Simple rate limiter for public HTTP calls."""

    def __init__(self, max_calls: int = 40, period_seconds: float = 5.0):
        self.max_calls = int(max_calls)
        self.period = float(period_seconds)
        self.calls: List[float] = []

    def wait(self) -> None:
        now = time.time()
        cutoff = now - self.period
        self.calls = [t for t in self.calls if t >= cutoff]
        if len(self.calls) >= self.max_calls:
            sleep_for = (self.calls[0] + self.period) - now
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.calls.append(time.time())


# =============================
# Bybit OHLCV Downloader (Public API v5)
# =============================

BYBIT_BASE = "https://api.bybit.com"


def _to_ms(ts: pd.Timestamp) -> int:
    ts_val = pd.Timestamp(ts)
    if ts_val.tzinfo is None:
        ts_val = ts_val.tz_localize("UTC")
    else:
        ts_val = ts_val.tz_convert("UTC")
    return int(ts_val.value // 1_000_000)


def bybit_get_klines(
    symbol: str,
    category: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    session: requests.Session,
    rl: RateLimiter,
    max_retries: int = 8,
) -> List[List[str]]:
    """
    GET /v5/market/kline
    Returns list rows: [startTime, open, high, low, close, volume, turnover]
    """
    url = BYBIT_BASE + "/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": int(start_ms),
        "end": int(end_ms),
        "limit": int(limit),
    }

    for attempt in range(int(max_retries)):
        rl.wait()
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 403:
                time.sleep(min(60, (2 ** attempt)) + np.random.random())
                continue
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") != 0:
                time.sleep(min(30, (2 ** attempt)) + np.random.random())
                continue
            rows = (data.get("result") or {}).get("list") or []
            return rows
        except Exception:
            time.sleep(min(30, (2 ** attempt)) + np.random.random())
    raise RuntimeError(f"Bybit API failed repeatedly for {symbol} {category} {interval}")


def fetch_ohlcv_bybit(
    symbol: str,
    category: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit_per_call: int = 200,
) -> pd.DataFrame:
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    rl = RateLimiter(max_calls=40, period_seconds=5.0)
    sess = requests.Session()

    all_rows: List[List[str]] = []
    cursor = start_ms

    while True:
        rows = bybit_get_klines(
            symbol=symbol,
            category=category,
            interval=interval,
            start_ms=cursor,
            end_ms=end_ms,
            limit=limit_per_call,
            session=sess,
            rl=rl,
        )
        if not rows:
            break
        rows = rows[::-1]  # newest-first -> oldest-first
        all_rows.extend(rows)
        last_ts = int(rows[-1][0])
        next_cursor = last_ts + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < limit_per_call:
            break

    if not all_rows:
        raise RuntimeError(f"No OHLCV returned for {symbol} {category} {interval}")

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime([int(r[0]) for r in all_rows], unit="ms", utc=True),
            "open": [float(r[1]) for r in all_rows],
            "high": [float(r[2]) for r in all_rows],
            "low": [float(r[3]) for r in all_rows],
            "close": [float(r[4]) for r in all_rows],
            "volume": [float(r[5]) for r in all_rows],
        }
    )
    return df.drop_duplicates(subset=["ts"]).sort_values("ts").set_index("ts")


def load_or_fetch_ohlcv(
    symbol: str,
    category: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str = "./bybit_cache",
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"{symbol}_{category}_{interval}_{start.date()}_{end.date()}.csv"
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["ts"])
        return df.set_index("ts").sort_index()
    print(f"Downloading {symbol} {category} {interval} from Bybit (public API)...")
    df = fetch_ohlcv_bybit(symbol, category, interval, start, end)
    df.reset_index().to_csv(path, index=False)
    print("Saved cache:", path)
    return df


# =============================
# Strategy Helpers
# =============================


def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's RMA (smoothed moving average) with SMA seed over the first length values.
    """
    values = series.astype(float).values
    out = np.full_like(values, np.nan, dtype=float)
    if len(values) < length:
        return pd.Series(out, index=series.index)
    out[length - 1] = np.mean(values[:length])
    alpha = 1.0 / float(length)
    for i in range(length, len(values)):
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fees: float
    reason: str


@dataclass
class BacktestResult:
    final_equity: float
    total_pnl: float
    win_rate: float
    num_trades: int
    trades: List[Dict[str, Any]]
    equity_curve: pd.DataFrame


# =============================
# Strategy Computation
# =============================


def compute_grid_levels(df: pd.DataFrame, band_mult: float = 2.5) -> Dict[str, List[pd.Series]]:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    main_rma = rma(close, 100)

    premium_zones = []
    discount_zones = []
    for k in range(1, 9):
        factor_up = 1 + band_mult * 0.01 * k  # band_mult percent per level
        factor_dn = 1 - band_mult * 0.01 * k
        premium_zones.append(ema(main_rma * factor_up, 5))
        discount_zones.append(ema(main_rma * factor_dn, 5))

    cover_main = ema(main_rma, 1)  # same as main_rma but kept for uniformity

    entry_conditions = [crossover(high, pz) for pz in premium_zones]
    exit_conditions = [crossunder(low, dz) for dz in discount_zones]
    exit_conditions.append(crossunder(low, cover_main))  # coverCondition9

    return {
        "entry_conditions": entry_conditions,
        "exit_conditions": exit_conditions,
        "main_rma": main_rma,
        "high": high,
        "low": low,
        "close": close,
    }


def run_backtest(
    df: pd.DataFrame,
    commission_rate: float = 0.0001,
    slippage_bps: float = 0.0,
    initial_equity: float = 1000.0,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> BacktestResult:
    """
    Executes the short-only 5m strategy with fixed 1% TP and fallback exits.
    """
    if start_time is None:
        start_time = pd.Timestamp("2024-01-01", tz="UTC")
    if end_time is None:
        end_time = pd.Timestamp("2027-01-01", tz="UTC")

    in_date = (df.index >= start_time) & (df.index <= end_time)
    levels = compute_grid_levels(df)

    entry_conditions = levels["entry_conditions"]
    exit_conditions = levels["exit_conditions"]
    close = levels["close"]
    high = levels["high"]
    low = levels["low"]

    # Build entrySignal with priority 8 -> 1
    entry_signal = pd.Series(0, index=df.index, dtype=int)
    for k in range(8, 0, -1):
        mask = entry_conditions[k - 1] & (entry_signal == 0)
        entry_signal.loc[mask] = k

    exit_signal = exit_conditions[0]
    for cond in exit_conditions[1:]:
        exit_signal |= cond

    equity = float(initial_equity)
    equity_curve = []
    trades: List[Trade] = []

    position_qty = 0.0
    entry_price = 0.0
    entry_fee = 0.0
    tp_pct = 0.01
    slip = slippage_bps / 1e4

    for i in range(1, len(df)):
        ts = df.index[i]
        price_close = float(close.iloc[i])
        price_high = float(high.iloc[i])
        price_low = float(low.iloc[i])

        # mark-to-market for curve
        unrealized = 0.0
        if position_qty != 0.0:
            # short PnL: (entry - current) * qty
            unrealized = (entry_price - price_close) * abs(position_qty)
        equity_curve.append({"ts": ts, "equity": equity + unrealized})

        # manage open position first
        if position_qty != 0.0:
            tp_price = entry_price * (1 - tp_pct)
            hit_tp = price_low <= tp_price
            hit_exit = bool(exit_signal.iloc[i] and in_date.iloc[i])

            exit_px = None
            reason = None
            if hit_tp:
                exit_px = tp_price
                reason = "TP"
            elif hit_exit:
                exit_px = price_close * (1 - slip)  # buying to cover, slip improves buy slightly
                reason = "EXIT_SIGNAL"

            if exit_px is not None:
                notional = abs(position_qty) * exit_px
                exit_fee = notional * commission_rate
                pnl = (entry_price - exit_px) * abs(position_qty)
                equity += pnl
                equity -= exit_fee
                trades.append(
                    Trade(
                        entry_ts=entry_time,
                        exit_ts=ts,
                        entry_price=entry_price,
                        exit_price=exit_px,
                        qty=-abs(position_qty),
                        pnl=pnl,
                        fees=entry_fee + exit_fee,
                        reason=reason,
                    )
                )
                position_qty = 0.0
                entry_price = 0.0
                entry_fee = 0.0
            # skip entries on same bar after exit to mimic single action per bar
            continue

        # flat: look for entry
        if not in_date.iloc[i]:
            continue
        if entry_signal.iloc[i] != 0:
            exec_price = price_close * (1 - slip)  # selling to open short, slip worsens price slightly
            notional = equity  # 100% of equity
            if notional <= 0:
                continue
            qty = notional / exec_price
            fee = notional * commission_rate
            equity -= fee
            position_qty = -qty
            entry_price = exec_price
            entry_fee = fee
            entry_time = ts

    # Close at end if still open
    if position_qty != 0.0:
        ts = df.index[-1]
        exit_px = float(close.iloc[-1]) * (1 - slip)
        notional = abs(position_qty) * exit_px
        exit_fee = notional * commission_rate
        pnl = (entry_price - exit_px) * abs(position_qty)
        equity += pnl
        equity -= exit_fee
        trades.append(
            Trade(
                entry_ts=entry_time,
                exit_ts=ts,
                entry_price=entry_price,
                exit_price=exit_px,
                qty=-abs(position_qty),
                pnl=pnl,
                fees=entry_fee + exit_fee,
                reason="EOD_CLOSE",
            )
        )
        equity_curve.append({"ts": ts, "equity": equity})

    equity_df = pd.DataFrame(equity_curve)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    win_rate = wins / max(1, wins + losses)

    return BacktestResult(
        final_equity=equity,
        total_pnl=equity - initial_equity,
        win_rate=win_rate,
        num_trades=len(trades),
        trades=[asdict(t) for t in trades],
        equity_curve=equity_df,
    )


# =============================
# CLI
# =============================


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-only 5m grid strategy backtester (Python translation of Pine).")
    parser.add_argument("--symbol", default="SOLUSDT", help="Instrument symbol (e.g., SOLUSDT).")
    parser.add_argument("--category", default="linear", help="Bybit category: linear, inverse, spot.")
    parser.add_argument("--start", default="2024-01-01", help="Start date (UTC, ISO format).")
    parser.add_argument("--end", default="2027-01-01", help="End date (UTC, ISO format).")
    parser.add_argument("--initial-equity", type=float, default=1000.0, help="Starting equity.")
    parser.add_argument("--commission-rate", type=float, default=0.0001, help="Commission rate (percent=0.01% default).")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage in basis points (per side).")
    parser.add_argument("--cache-dir", default="./bybit_cache", help="Cache directory for OHLCV CSVs.")
    args = parser.parse_args()

    interval = "5"
    if interval not in {"5", "5m", "5M"}:
        raise ValueError("This strategy is HARD-CODED to run on 5-minute candles only.")

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    df = load_or_fetch_ohlcv(
        symbol=args.symbol,
        category=args.category,
        interval=interval,
        start=start_ts,
        end=end_ts,
        cache_dir=args.cache_dir,
    )
    print(f"Data rows: {len(df)}  |  {args.symbol} {args.category} {interval}m  |  {df.index.min()} -> {df.index.max()}")

    res = run_backtest(
        df=df,
        commission_rate=float(args.commission_rate),
        slippage_bps=float(args.slippage_bps),
        initial_equity=float(args.initial_equity),
        start_time=start_ts,
        end_time=end_ts,
    )

    print("\n=== Backtest Results ===")
    print(f"Final equity: {res.final_equity:.2f}")
    print(f"Total PnL: {res.total_pnl:.2f}")
    print(f"Win rate: {res.win_rate*100:.2f}%")
    print(f"Trades: {res.num_trades}")

    # Save outputs
    os.makedirs("results", exist_ok=True)
    res.equity_curve.to_csv("results/equity_curve.csv", index=False)
    pd.DataFrame(res.trades).to_csv("results/trades.csv", index=False)
    print("Saved results/equity_curve.csv and results/trades.csv")


if __name__ == "__main__":
    main()
