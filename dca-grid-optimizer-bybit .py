from __future__ import annotations

"""
Duncan-like Mean Reversion / DCA (Grid-ish) Backtester + Random Optimizer
- Pulls OHLCV from Bybit public API v5 (no keys)
- Backtests contrarian BBands+RSI entries + DCA ladder + basket TP
- Includes fees, spread, slippage, leverage, simplified liquidation
- Optimizer runs 1000 randomized configs with a 95% max "margin budget" constraint

DEFAULTS:
- Symbol: SOLUSDT
- Category: linear (USDT perps)
- Interval: 5m
- Window: last 90 days
"""

import os
import time
import json
import random
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple, Union


# =============================
# Rate Limiter
# =============================
class RateLimiter:
    """
    Conservative rate limiter for public endpoints.
    """
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

def _to_ms(ts: Union[str, int, float, pd.Timestamp]) -> int:
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 1_000_000)
    if isinstance(ts, (int, float, np.integer, np.floating)):
        if ts > 1e12:
            return int(ts)
        if ts > 1e9:
            return int(ts * 1000)
        return int(ts)
    dt = pd.to_datetime(ts, utc=True)
    return int(dt.value // 1_000_000)

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
    Calls: GET /v5/market/kline
    Returns list rows:
      [startTime, open, high, low, close, volume, turnover]
    Usually newest-first.
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

            # Bybit sometimes replies 403 "access too frequent"
            if r.status_code == 403:
                time.sleep(min(60, (2 ** attempt)) + random.random())
                continue

            r.raise_for_status()
            data = r.json()

            if data.get("retCode") != 0:
                time.sleep(min(30, (2 ** attempt)) + random.random())
                continue

            rows = (data.get("result") or {}).get("list") or []
            return rows

        except Exception:
            time.sleep(min(30, (2 ** attempt)) + random.random())

    raise RuntimeError(f"Bybit API failed repeatedly for {symbol} {category} {interval}")

def fetch_ohlcv_bybit(
    symbol: str,
    category: str,
    interval: str,
    start: Union[str, int, float, pd.Timestamp],
    end: Union[str, int, float, pd.Timestamp],
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

        # API often returns newest-first; reverse to oldest-first
        rows = rows[::-1]
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

    df = pd.DataFrame({
        "ts": pd.to_datetime([int(r[0]) for r in all_rows], unit="ms"),
        "open": [float(r[1]) for r in all_rows],
        "high": [float(r[2]) for r in all_rows],
        "low":  [float(r[3]) for r in all_rows],
        "close":[float(r[4]) for r in all_rows],
        "volume":[float(r[5]) for r in all_rows],
    })

    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").set_index("ts")
    return df

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
# Indicators
# =============================
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def bollinger(series: pd.Series, length: int = 20, stdev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(length).mean()
    sd = series.rolling(length).std(ddof=0)
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    return lower, ma, upper


# =============================
# Config / Data Structures
# =============================
@dataclass
class Config:
    initial_equity: float = 400.0
    leverage: float = 10.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    use_taker: bool = True

    spread_bps: float = 2.0
    slippage_bps: float = 1.0

    bb_len: int = 20
    bb_stdev: float = 2.0
    rsi_len: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    base_margin_frac: float = 0.03
    add_margin_frac: float = 0.03
    add_step_pct: float = 0.008
    max_adds: int = 10
    size_mult: float = 1.35

    basket_tp_pct: float = 0.006
    basket_time_stop_bars: Optional[int] = None

    equity_dd_stop_pct: Optional[float] = None
    maintenance_margin_rate: float = 0.005

    enable_long: bool = True
    enable_short: bool = True


@dataclass
class Fill:
    ts: Any
    side: str
    qty: float
    price: float
    fee: float
    reason: str


@dataclass
class Trade:
    entry_ts: Any
    exit_ts: Any
    side: str
    qty: float
    avg_entry: float
    avg_exit: float
    pnl: float
    pnl_pct_on_margin: float
    bars_held: int
    fills: List[Dict[str, Any]]


# =============================
# Execution / Pricing Helpers
# =============================
def apply_exec_price(mid_price: float, side: str, spread_bps: float, slippage_bps: float) -> float:
    spread = (spread_bps / 1e4) * mid_price
    slip = (slippage_bps / 1e4) * mid_price
    half = spread / 2.0
    if side == "BUY":
        return mid_price + half + slip
    else:
        return mid_price - half - slip

def fee_for_notional(notional: float, cfg: Config) -> float:
    rate = cfg.taker_fee if cfg.use_taker else cfg.maker_fee
    return abs(notional) * rate


# =============================
# Core Strategy Types
# =============================
class BasketPosition:
    def __init__(self, side: str):
        self.side = side  # "LONG" or "SHORT"
        self.qty = 0.0
        self.vwap = 0.0
        self.margin_used = 0.0
        self.add_count = 0
        self.entry_bar = None
        self.fills: List[Fill] = []

    def direction(self) -> int:
        return 1 if self.side == "LONG" else -1

    def notional(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        d = self.direction()
        return d * self.qty * (price - self.vwap)

    def adverse_move_pct(self, price: float) -> float:
        if self.qty <= 0 or self.vwap <= 0:
            return 0.0
        d = self.direction()
        if d == 1:
            return max(0.0, (self.vwap - price) / self.vwap)
        else:
            return max(0.0, (price - self.vwap) / self.vwap)

    def add_fill(self, fill: Fill):
        self.fills.append(fill)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) * -1.0  # positive fraction


# ---- chunk 1 ends here ----
# =============================
# Backtest Engine
# =============================
def backtest(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    df: DataFrame indexed by datetime 'ts', columns: open, high, low, close, volume
    """
    df = df.copy()

    # Validate columns
    cols = {c.lower(): c for c in df.columns}
    for need in ["open", "high", "low", "close"]:
        if need not in cols:
            raise ValueError(f"Data missing required column: {need}")

    # Build ts column from index if not present
    if "ts" not in df.columns:
        df["ts"] = df.index

    close = df[cols["close"]].astype(float)

    # Indicators
    bb_l, bb_m, bb_u = bollinger(close, cfg.bb_len, cfg.bb_stdev)
    rs = rsi(close, cfg.rsi_len)

    equity = cfg.initial_equity
    start_equity = equity
    peak_equity = equity

    pos: Optional[BasketPosition] = None
    trades: List[Trade] = []
    equity_curve = []

    def liquidated(price: float, pos: BasketPosition) -> bool:
        notional = abs(pos.notional(price))
        maint = notional * cfg.maintenance_margin_rate
        equity_in_pos = pos.margin_used + pos.unrealized_pnl(price)
        return equity_in_pos <= maint

    def open_position(side: str, i: int, mid: float):
        nonlocal equity, pos
        pos = BasketPosition(side=side)
        pos.entry_bar = i

        margin = equity * cfg.base_margin_frac
        margin = min(margin, equity)
        if margin <= 0:
            pos = None
            return

        notional = margin * cfg.leverage
        fill_side = "BUY" if side == "LONG" else "SELL"
        px = apply_exec_price(mid, fill_side, cfg.spread_bps, cfg.slippage_bps)
        qty = notional / px

        fee = fee_for_notional(notional, cfg)
        equity -= fee
        equity -= margin
        pos.margin_used += margin

        pos.qty = qty
        pos.vwap = px
        pos.add_fill(Fill(df["ts"].iloc[i], fill_side, qty, px, fee, "ENTRY"))

    def add_to_position(i: int, mid: float):
        nonlocal equity, pos
        assert pos is not None
        if pos.add_count >= cfg.max_adds:
            return

        adverse = pos.adverse_move_pct(mid)
        next_step = cfg.add_step_pct * (pos.add_count + 1)
        if adverse < next_step:
            return

        add_margin = equity * cfg.add_margin_frac * (cfg.size_mult ** pos.add_count)
        add_margin = min(add_margin, equity)
        if add_margin <= 0:
            return

        notional = add_margin * cfg.leverage
        fill_side = "BUY" if pos.side == "LONG" else "SELL"
        px = apply_exec_price(mid, fill_side, cfg.spread_bps, cfg.slippage_bps)
        qty = notional / px

        fee = fee_for_notional(notional, cfg)
        equity -= fee
        equity -= add_margin
        pos.margin_used += add_margin

        new_qty = pos.qty + qty
        pos.vwap = (pos.vwap * pos.qty + px * qty) / new_qty
        pos.qty = new_qty
        pos.add_count += 1

        pos.add_fill(Fill(df["ts"].iloc[i], fill_side, qty, px, fee, f"ADD_{pos.add_count}"))

    def should_take_profit(mid: float) -> bool:
        assert pos is not None
        pnl = pos.unrealized_pnl(mid)
        notional = abs(pos.notional(mid))
        if notional <= 0:
            return False
        return (pnl / notional) >= cfg.basket_tp_pct

    def should_time_stop(i: int) -> bool:
        if cfg.basket_time_stop_bars is None:
            return False
        assert pos is not None
        return (i - pos.entry_bar) >= cfg.basket_time_stop_bars

    def close_position(i: int, mid: float, reason: str):
        nonlocal equity, pos, trades
        assert pos is not None

        fill_side = "SELL" if pos.side == "LONG" else "BUY"
        px = apply_exec_price(mid, fill_side, cfg.spread_bps, cfg.slippage_bps)

        notional = abs(pos.qty * px)
        fee = fee_for_notional(notional, cfg)
        equity -= fee

        pnl = pos.unrealized_pnl(px)
        equity += pos.margin_used + pnl

        fills_dict = [asdict(f) for f in pos.fills]
        fills_dict.append(asdict(Fill(df["ts"].iloc[i], fill_side, pos.qty, px, fee, reason)))

        margin_total = pos.margin_used if pos.margin_used > 0 else 1e-9
        trades.append(
            Trade(
                entry_ts=pos.fills[0].ts if pos.fills else df["ts"].iloc[i],
                exit_ts=df["ts"].iloc[i],
                side=pos.side,
                qty=pos.qty,
                avg_entry=pos.vwap,
                avg_exit=px,
                pnl=pnl - sum(f.fee for f in pos.fills) - fee,
                pnl_pct_on_margin=(pnl / margin_total),
                bars_held=(i - (pos.entry_bar if pos.entry_bar is not None else i)),
                fills=fills_dict
            )
        )
        pos = None

    # -----------------------------
    # Main loop
    # -----------------------------
    for i in range(len(df)):
        mid = float(close.iloc[i])
        ts = df["ts"].iloc[i]

        # equity dd stop from start (optional)
        if cfg.equity_dd_stop_pct is not None:
            if equity <= start_equity * (1.0 - cfg.equity_dd_stop_pct):
                if pos is not None:
                    close_position(i, mid, "DD_STOP")
                break

        # mark-to-market equity
        m2m_equity = equity
        if pos is not None:
            m2m_equity += pos.margin_used + pos.unrealized_pnl(mid)
        peak_equity = max(peak_equity, m2m_equity)

        equity_curve.append({"ts": ts, "equity": m2m_equity, "cash": equity, "in_pos": pos is not None})

        # Manage open position
        if pos is not None:
            if liquidated(mid, pos):
                close_position(i, mid, "LIQUIDATED")
                continue

            add_to_position(i, mid)

            if should_take_profit(mid):
                close_position(i, mid, "BASKET_TP")
                continue
            if should_time_stop(i):
                close_position(i, mid, "TIME_STOP")
                continue

            continue

        # Entry logic (contrarian)
        if i < max(cfg.bb_len, cfg.rsi_len):
            continue

        # Long entry
        if cfg.enable_long and (mid < bb_l.iloc[i]) and (rs.iloc[i] <= cfg.rsi_oversold):
            open_position("LONG", i, mid)
            continue

        # Short entry
        if cfg.enable_short and (mid > bb_u.iloc[i]) and (rs.iloc[i] >= cfg.rsi_overbought):
            open_position("SHORT", i, mid)
            continue

    # Close at end if still open
    if pos is not None:
        close_position(len(df) - 1, float(close.iloc[-1]), "EOD_CLOSE")

    eq_df = pd.DataFrame(equity_curve)
    final_equity = float(eq_df["equity"].iloc[-1]) if len(eq_df) else equity

    pnl_total = final_equity - cfg.initial_equity
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    win_rate = wins / max(1, (wins + losses))
    max_dd = max_drawdown(eq_df["equity"]) if len(eq_df) else 0.0

    return {
        "config": asdict(cfg),
        "final_equity": final_equity,
        "total_pnl": pnl_total,
        "total_pnl_pct": pnl_total / cfg.initial_equity if cfg.initial_equity > 0 else 0.0,
        "num_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "max_drawdown_pct": max_dd,
        "equity_curve": eq_df,
        "trades": [asdict(t) for t in trades],
    }

# ---- chunk 2 ends here ----
# =============================
# Optimizer Helpers
# =============================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _rand_log_uniform(lo: float, hi: float) -> float:
    """
    log-uniform random sample between lo and hi (both > 0)
    """
    lo = float(lo)
    hi = float(hi)
    if lo <= 0 or hi <= 0 or lo >= hi:
        return float(lo)
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))

def _rand_choice_int(values: List[int]) -> int:
    return int(random.choice(values))

def margin_budget_ok(cfg: Config, budget_frac: float = 0.95) -> bool:
    """
    Constraint: worst-case total margin used (entry + all adds) must be <= budget_frac of starting equity.
    We approximate using fractions of equity at time of add (which actually declines), so this is conservative.
    """
    # Worst-case planned margin fractions (ignoring equity decay, so over-estimate)
    planned = cfg.base_margin_frac
    # each add uses add_margin_frac * size_mult^k for k=0..max_adds-1
    add_sum = 0.0
    for k in range(int(cfg.max_adds)):
        add_sum += cfg.add_margin_frac * (cfg.size_mult ** k)
    planned += add_sum
    return planned <= float(budget_frac)

def sample_random_config(base: Config, budget_frac: float = 0.95) -> Config:
    """
    Randomize parameters around plausible ranges.
    Ensures: margin budget <= budget_frac via resampling.
    """
    # Try a bunch of times to satisfy the budget constraint
    for _ in range(200):
        cfg = Config(**asdict(base))

        # Indicator params
        cfg.bb_len = _rand_choice_int([14, 16, 18, 20, 22, 24, 26, 28, 30])
        cfg.bb_stdev = float(np.random.uniform(1.6, 2.8))
        cfg.rsi_len = _rand_choice_int([7, 9, 10, 12, 14, 16, 18, 21])
        # Oversold / overbought: keep separation
        cfg.rsi_oversold = float(np.random.uniform(15.0, 40.0))
        cfg.rsi_overbought = float(np.random.uniform(60.0, 85.0))
        if cfg.rsi_overbought <= cfg.rsi_oversold + 10:
            cfg.rsi_overbought = cfg.rsi_oversold + 20.0
            cfg.rsi_overbought = _clamp(cfg.rsi_overbought, 55.0, 90.0)

        # DCA ladder
        cfg.max_adds = _rand_choice_int([4, 5, 6, 7, 8, 9, 10, 12, 14])
        cfg.add_step_pct = float(_rand_log_uniform(0.002, 0.02))      # 0.2% .. 2.0%
        cfg.size_mult = float(np.random.uniform(1.0, 1.8))            # 1.0 .. 1.8

        # Margin fractions: base + add
        # Keep within reasonable
        cfg.base_margin_frac = float(_rand_log_uniform(0.005, 0.06))  # 0.5% .. 6%
        cfg.add_margin_frac = float(_rand_log_uniform(0.005, 0.06))   # 0.5% .. 6%

        # Basket TP: small mean-reversion exits
        cfg.basket_tp_pct = float(_rand_log_uniform(0.0015, 0.02))    # 0.15% .. 2.0%

        # Optional time stop sometimes
        if random.random() < 0.25:
            cfg.basket_time_stop_bars = _rand_choice_int([120, 180, 240, 300, 360, 480])
        else:
            cfg.basket_time_stop_bars = None

        # Sides
        side_mode = random.random()
        if side_mode < 0.15:
            cfg.enable_long = True
            cfg.enable_short = False
        elif side_mode < 0.30:
            cfg.enable_long = False
            cfg.enable_short = True
        else:
            cfg.enable_long = True
            cfg.enable_short = True

        # Keep fees/slippage same as base (already in engine)
        cfg.use_taker = base.use_taker
        cfg.maker_fee = base.maker_fee
        cfg.taker_fee = base.taker_fee
        cfg.spread_bps = base.spread_bps
        cfg.slippage_bps = base.slippage_bps
        cfg.leverage = base.leverage
        cfg.initial_equity = base.initial_equity
        cfg.maintenance_margin_rate = base.maintenance_margin_rate

        # Optional kill switch: keep None (Duncan-ish), but allow rare dd stop
        if random.random() < 0.10:
            cfg.equity_dd_stop_pct = float(np.random.uniform(0.6, 0.95))
        else:
            cfg.equity_dd_stop_pct = None

        if margin_budget_ok(cfg, budget_frac=budget_frac):
            return cfg

    # If we fail to sample within budget after many tries, force a safe config
    cfg = Config(**asdict(base))
    cfg.base_margin_frac = min(cfg.base_margin_frac, 0.02)
    cfg.add_margin_frac = min(cfg.add_margin_frac, 0.02)
    cfg.size_mult = min(cfg.size_mult, 1.2)
    cfg.max_adds = min(cfg.max_adds, 8)
    return cfg

def score_result(res: Dict[str, Any]) -> float:
    """
    Scoring:
    - Prefer higher final equity
    - Penalize drawdown
    - Penalize liquidation count if any appear in trades (reason == LIQUIDATED)
    """
    final_eq = float(res.get("final_equity", 0.0))
    dd = float(res.get("max_drawdown_pct", 0.0))

    # liquidation detection from trade fills
    liq = 0
    for t in res.get("trades", []) or []:
        fills = t.get("fills", []) or []
        for f in fills:
            if str(f.get("reason", "")).upper() == "LIQUIDATED":
                liq += 1

    # If any liquidation, nuke score (you can change this later)
    if liq > 0:
        return -1e18

    # Drawdown penalty: scale by equity so score units match
    # Score = final_eq - (dd_penalty * final_eq)
    dd_penalty = 1.25 * dd  # harsher penalty than 1:1
    return final_eq * (1.0 - dd_penalty)

def run_optimizer(
    df: pd.DataFrame,
    base_cfg: Config,
    n_trials: int = 1000,
    budget_frac: float = 0.95,
    keep_top: int = 20,
    seed: Optional[int] = 1337,
) -> List[Dict[str, Any]]:
    """
    Runs random search and returns list of best result dicts (with config + stats).
    """
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    best: List[Dict[str, Any]] = []

    for idx in range(1, int(n_trials) + 1):
        cfg = sample_random_config(base_cfg, budget_frac=budget_frac)

        try:
            res = backtest(df, cfg)
        except Exception as e:
            # Skip broken configs (should be rare)
            continue

        s = score_result(res)

        row = {
            "score": float(s),
            "final_equity": float(res["final_equity"]),
            "total_pnl": float(res["total_pnl"]),
            "total_pnl_pct": float(res["total_pnl_pct"]),
            "num_trades": int(res["num_trades"]),
            "win_rate": float(res["win_rate"]),
            "max_drawdown_pct": float(res["max_drawdown_pct"]),
            "config": res["config"],
        }

        best.append(row)
        best.sort(key=lambda x: x["score"], reverse=True)
        if len(best) > int(keep_top):
            best = best[:int(keep_top)]

        # Progress print every 50 runs
        if idx % 50 == 0:
            top = best[0] if best else None
            if top:
                print(
                    f"[{idx}/{n_trials}] best score={top['score']:.2f} "
                    f"final_eq={top['final_equity']:.2f} dd={top['max_drawdown_pct']*100:.2f}% "
                    f"trades={top['num_trades']} win={top['win_rate']*100:.1f}%"
                )

    return best

# ---- chunk 3 ends here ----
# =============================
# Main Runner
# =============================
def save_best_outputs(best: List[Dict[str, Any]], out_dir: str = ".") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save leaderboard CSV
    rows = []
    for b in best:
        c = b["config"]
        flat = {
            "score": b["score"],
            "final_equity": b["final_equity"],
            "total_pnl": b["total_pnl"],
            "total_pnl_pct": b["total_pnl_pct"],
            "num_trades": b["num_trades"],
            "win_rate": b["win_rate"],
            "max_drawdown_pct": b["max_drawdown_pct"],

            # Core params
            "bb_len": c["bb_len"],
            "bb_stdev": c["bb_stdev"],
            "rsi_len": c["rsi_len"],
            "rsi_overbought": c["rsi_overbought"],
            "rsi_oversold": c["rsi_oversold"],

            "base_margin_frac": c["base_margin_frac"],
            "add_margin_frac": c["add_margin_frac"],
            "add_step_pct": c["add_step_pct"],
            "max_adds": c["max_adds"],
            "size_mult": c["size_mult"],

            "basket_tp_pct": c["basket_tp_pct"],
            "basket_time_stop_bars": c["basket_time_stop_bars"],

            "enable_long": c["enable_long"],
            "enable_short": c["enable_short"],

            # Execution costs
            "use_taker": c["use_taker"],
            "maker_fee": c["maker_fee"],
            "taker_fee": c["taker_fee"],
            "spread_bps": c["spread_bps"],
            "slippage_bps": c["slippage_bps"],

            # Risk
            "equity_dd_stop_pct": c["equity_dd_stop_pct"],
            "maintenance_margin_rate": c["maintenance_margin_rate"],
            "leverage": c["leverage"],
        }
        rows.append(flat)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "best_results.csv")
    df.to_csv(csv_path, index=False)

    # Save best config JSON
    if best:
        best_cfg = best[0]["config"]
        json_path = os.path.join(out_dir, "best_config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=2)

def run_best_and_export(df: pd.DataFrame, best_cfg_dict: Dict[str, Any], out_dir: str = ".") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config(**best_cfg_dict)
    res = backtest(df, cfg)

    # Save equity and trades
    eq_path = os.path.join(out_dir, "equity_curve_best.csv")
    tr_path = os.path.join(out_dir, "trades_best.csv")
    res["equity_curve"].to_csv(eq_path, index=False)
    pd.DataFrame(res["trades"]).to_csv(tr_path, index=False)

    return res

if __name__ == "__main__":
    # -----------------------------
    # Base / Defaults
    # -----------------------------
    base_cfg = Config(
        initial_equity=400.0,
        leverage=10.0,

        # fees + execution costs (already modeled in engine)
        maker_fee=0.0002,
        taker_fee=0.0006,
        use_taker=True,
        spread_bps=2.0,
        slippage_bps=1.0,

        # keep Duncan-ish: no stop by default
        equity_dd_stop_pct=None,

        maintenance_margin_rate=0.005,
    )

    # -----------------------------
    # Market / Data Settings
    # -----------------------------
    symbol = "SOLUSDT"
    category = "linear"   # USDT perps
    interval = "5"        # 5m candles

    # last 90 days window
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=90)

    df = load_or_fetch_ohlcv(symbol, category, interval, start, end)

    # -----------------------------
    # Optimizer Settings
    # -----------------------------
    N_TRIALS = 1000
    KEEP_TOP = 20
    BUDGET_FRAC = 0.95
    SEED = 1337

    print(f"Data rows: {len(df)}  |  {symbol} {category} {interval}  |  {start} -> {end}")
    print(f"Running optimizer: trials={N_TRIALS}, keep_top={KEEP_TOP}, margin_budget={BUDGET_FRAC*100:.0f}%")

    best = run_optimizer(
        df=df,
        base_cfg=base_cfg,
        n_trials=N_TRIALS,
        budget_frac=BUDGET_FRAC,
        keep_top=KEEP_TOP,
        seed=SEED,
    )

    if not best:
        raise RuntimeError("No valid results produced. Try fewer constraints or check data.")

    # Save summary outputs
    save_best_outputs(best, out_dir=".")

    # Print top 5 to console
    print("\nTOP 5 RESULTS:")
    for i, b in enumerate(best[:5], start=1):
        c = b["config"]
        print(
            f"{i}) score={b['score']:.2f} final_eq={b['final_equity']:.2f} "
            f"dd={b['max_drawdown_pct']*100:.2f}% trades={b['num_trades']} win={b['win_rate']*100:.1f}% "
            f"tp={c['basket_tp_pct']:.4f} step={c['add_step_pct']:.4f} mult={c['size_mult']:.2f} "
            f"adds={c['max_adds']} base={c['base_margin_frac']:.4f} add={c['add_margin_frac']:.4f} "
            f"bb=({c['bb_len']},{c['bb_stdev']:.2f}) rsi=({c['rsi_len']},{c['rsi_oversold']:.1f}/{c['rsi_overbought']:.1f}) "
            f"sides={'LS' if (c['enable_long'] and c['enable_short']) else ('L' if c['enable_long'] else 'S')}"
        )

    # Run best config and export detailed files
    best_cfg_dict = best[0]["config"]
    best_res = run_best_and_export(df, best_cfg_dict, out_dir=".")

    print("\nBEST CONFIG RUN (exported equity_curve_best.csv + trades_best.csv):")
    print("Final equity:", round(best_res["final_equity"], 2))
    print("Total PnL:", round(best_res["total_pnl"], 2), f"({best_res['total_pnl_pct']*100:.2f}%)")
    print("Trades:", best_res["num_trades"], "Win rate:", f"{best_res['win_rate']*100:.2f}%")
    print("Max DD:", f"{best_res['max_drawdown_pct']*100:.2f}%")

    print("\nSaved:")
    print("- best_results.csv")
    print("- best_config.json")
    print("- equity_curve_best.csv")
    print("- trades_best.csv")

# ---- chunk 4 ends here ----
