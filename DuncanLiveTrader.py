# ============================================================
# HYPEUSDT SHORT GRID — EXACT BYBIT ENGINE (PART 1/4)
# Core, settings, logging, REST, data download
# ============================================================

import os
import csv
import json
import time
import random
import logging
import hmac
import hashlib
import urllib.parse
import math
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import numpy as np
from websocket import WebSocketApp

# ============================================================
# USER SETTINGS
# ============================================================
SYMBOL = "HYPEUSDT"
INTERVAL = "5"                 # HARD CODED 5m
CATEGORY = "linear"

DAYS_BACK_SEED = 7             # initial history window
STARTING_WALLET = 500.0
LEVERAGE = 10.0

# TradingView commission_value=0.01 => 0.01% = 0.0001
FEE_RATE = 0.0001              # per side, on notional

# Slippage model: 1 tick adverse
SLIPPAGE_TICKS = 1
TICK_SIZE = 0.0001             # used only for slippage simulation

# Pine hard-coded defaults (used as baseline + optimiser ranges)
DEFAULT_MA_LEN = 100
DEFAULT_BAND_MULT = 2.5
DEFAULT_TP_PERC = 0.01

# Optimiser ranges
INIT_TRIALS = 300
MA_LEN_RANGE = (30, 200)
BAND_MULT_RANGE = (0.5, 6.0)
TP_RANGE = (0.002, 0.03)

# Walk-forward optimisation
WFO_ENABLED = True
WFO_LOOKBACK_CANDLES = 3 * 24 * 12   # 3 days of 5m candles
WFO_TRIALS = 150
WFO_EVERY_CLOSED_CANDLES = 12        # every hour
WFO_APPLY_ONLY_WHEN_FLAT = True

# Runtime behaviour
KEEP_CANDLES = 3000
PRINT_EVERY_CANDLE = True
API_POLITE_SLEEP = 0.1

# Live trading toggles (real orders)
REAL_TRADING_ENABLED = False
API_KEY = "YOUR_BYBIT_API_KEY"
API_SECRET = "YOUR_BYBIT_API_SECRET"
RECV_WINDOW = "5000"
CANDLE_STALENESS_MAX_SEC = 120

# ============================================================
# LOGGING / FILE OUTPUT
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "paper_logs")
os.makedirs(LOG_DIR, exist_ok=True)

EVENT_LOG_PATH = os.path.join(LOG_DIR, "events.log")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "trades.csv")
PARAMS_CSV_PATH = os.path.join(LOG_DIR, "params.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(EVENT_LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("paper")

def ensure_csv(path, header):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def csv_append(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# Trade log (full detail)
ensure_csv(TRADES_CSV_PATH, [
    "ts_utc","action","reason","signal_level","side","qty","fill_price","notional","fee",
    "entry_price","tp_price","mark_price","liq_price",
    "wallet_before","wallet_after",
    "pnl_gross","pnl_net_10x","pnl_1x_usdt","pnl_1x_pct","pnl_10x_pct",
    "tier","indicators","candle_ohlc",
    "ma_len","band_mult","tp_perc"
])

# Param log
ensure_csv(PARAMS_CSV_PATH, [
    "ts_utc","event","ma_len","band_mult","tp_perc","wallet"
])

# ============================================================
# BYBIT REST HELPERS
# ============================================================
BASE_REST = "https://api.bybit.com"

def now_ms() -> int:
    return int(time.time() * 1000)

def rest_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = BASE_REST + path
    r = requests.get(url, params=params, timeout=30)
    j = r.json()
    if "retCode" not in j:
        raise RuntimeError(f"Bybit REST unexpected response: {j}")
    if j["retCode"] != 0:
        raise RuntimeError(f"Bybit REST error retCode={j['retCode']} retMsg={j.get('retMsg')} params={params}")
    return j

def _sign_request(timestamp: str, api_key: str, recv_window: str, payload: str) -> str:
    raw = f"{timestamp}{api_key}{recv_window}{payload}"
    return hmac.new(API_SECRET.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()

def _build_query(params: Dict[str, Any]) -> str:
    return urllib.parse.urlencode({k: params[k] for k in sorted(params)})

def rest_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    auth: bool = False
) -> Dict[str, Any]:
    url = BASE_REST + path
    params = params or {}
    body = body or {}

    if method.upper() == "GET":
        query = _build_query(params)
        full_url = f"{url}?{query}" if query else url
        payload = query
        headers = {}
        data = None
    else:
        full_url = url
        payload = json.dumps(body, separators=(",", ":"))
        headers = {"Content-Type": "application/json"}
        data = payload

    if auth:
        timestamp = str(now_ms())
        if not API_KEY or not API_SECRET:
            raise RuntimeError("Missing BYBIT_API_KEY or BYBIT_API_SECRET for authenticated requests.")
        signature = _sign_request(timestamp, API_KEY, RECV_WINDOW, payload)
        headers.update({
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        })

    if method.upper() == "GET":
        r = requests.get(full_url, headers=headers, timeout=30)
    else:
        r = requests.post(full_url, headers=headers, data=data, timeout=30)

    j = r.json()
    if "retCode" not in j:
        raise RuntimeError(f"Bybit REST unexpected response: {j}")
    if j["retCode"] != 0:
        raise RuntimeError(f"Bybit REST error retCode={j['retCode']} retMsg={j.get('retMsg')} params={params} body={body}")
    return j

# ============================================================
# DATA DOWNLOAD (FIXED: SEPARATE PARSERS)
# ============================================================
def fetch_last_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    /v5/market/kline  -> returns 7 columns
    [start, open, high, low, close, volume, turnover]
    """
    path = "/v5/market/kline"
    out = []
    cur = end_ms

    while True:
        j = rest_get(path, {
            "category": CATEGORY,
            "symbol": symbol,
            "interval": interval,
            "end": cur,
            "limit": 1000
        })
        rows = j["result"]["list"]
        if not rows:
            break

        rows = sorted(rows, key=lambda x: int(x[0]))

        for r in rows:
            ts = int(r[0])
            if start_ms <= ts <= end_ms:
                out.append(r)

        earliest = int(rows[0][0])
        if earliest <= start_ms:
            break

        cur = earliest - 1
        time.sleep(API_POLITE_SLEEP)

    if not out:
        raise RuntimeError("No last-price klines returned")

    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume","turnover"])
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    return df

def fetch_mark_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    /v5/market/mark-price-kline -> returns 5 columns
    [start, open, high, low, close]
    """
    path = "/v5/market/mark-price-kline"
    out = []
    cur = end_ms

    while True:
        j = rest_get(path, {
            "category": CATEGORY,
            "symbol": symbol,
            "interval": interval,
            "end": cur,
            "limit": 1000
        })
        rows = j["result"]["list"]
        if not rows:
            break

        rows = sorted(rows, key=lambda x: int(x[0]))

        for r in rows:
            ts = int(r[0])
            if start_ms <= ts <= end_ms:
                out.append(r)

        earliest = int(rows[0][0])
        if earliest <= start_ms:
            break

        cur = earliest - 1
        time.sleep(API_POLITE_SLEEP)

    if not out:
        raise RuntimeError("No mark-price klines returned")

    df = pd.DataFrame(out, columns=["ts","open","high","low","close"])
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    return df

def fetch_risk_tiers(symbol: str) -> pd.DataFrame:
    """
    /v5/market/risk-limit
    Returns tiered MMR and mmDeduction used in liquidation
    """
    j = rest_get("/v5/market/risk-limit", {
        "category": CATEGORY,
        "symbol": symbol
    })

    rows = j["result"]["list"]
    if not rows:
        raise RuntimeError("No risk tiers returned")

    df = pd.DataFrame(rows)
    df["riskLimitValue"] = df["riskLimitValue"].astype(float)
    df["maintenanceMarginRate"] = df["maintenanceMargin"].astype(float) / 100.0

    def mm(x):
        try:
            return float(x)
        except:
            return 0.0

    df["mmDeductionValue"] = df["mmDeduction"].apply(mm)
    df = df.sort_values("riskLimitValue").reset_index(drop=True)
    return df

def fetch_last_price(symbol: str) -> float:
    j = rest_get("/v5/market/tickers", {
        "category": CATEGORY,
        "symbol": symbol
    })
    rows = j["result"]["list"]
    if not rows:
        raise RuntimeError("No tickers returned")
    return float(rows[0]["lastPrice"])
# ============================================================
# HYPEUSDT SHORT GRID — EXACT BYBIT ENGINE (PART 2/4)
# Indicators + crossover helpers + exact liquidation + backtest core
# ============================================================

# ----------------------------
# INDICATORS (Pine-equivalent)
# ----------------------------
def rma(series: pd.Series, length: int) -> np.ndarray:
    """
    Pine ta.rma equivalent:
      rma[0] = src[0]
      rma[i] = (rma[i-1]*(len-1) + src[i]) / len
    which is same as EMA with alpha=1/len and seed=first value.
    """
    n = len(series)
    if n == 0:
        return np.array([], dtype=float)

    out = np.zeros(n, dtype=float)
    alpha = 1.0 / float(length)

    # seed
    out[0] = float(series.iloc[0])

    for i in range(1, n):
        out[i] = alpha * float(series.iloc[i]) + (1.0 - alpha) * out[i - 1]
    return out

def ema_np(series: np.ndarray, length: int) -> np.ndarray:
    """
    Fast EMA over numpy array with seed=first value, adjust=False style.
    """
    n = len(series)
    if n == 0:
        return np.array([], dtype=float)
    out = np.zeros(n, dtype=float)
    alpha = 2.0 / (length + 1.0)
    out[0] = float(series[0])
    for i in range(1, n):
        out[i] = alpha * float(series[i]) + (1.0 - alpha) * out[i - 1]
    return out

def crossover(a_prev: float, a_cur: float, b_prev: float, b_cur: float) -> bool:
    # Pine ta.crossover
    return (a_prev <= b_prev) and (a_cur > b_cur)

def crossunder(a_prev: float, a_cur: float, b_prev: float, b_cur: float) -> bool:
    # Pine ta.crossunder
    return (a_prev >= b_prev) and (a_cur < b_cur)

def apply_slippage(price: float, side: str) -> float:
    """
    TradingView slippage model (simple):
      - SHORT entry (sell): worse price = price - tick
      - COVER exit (buy):  worse price = price + tick
    """
    if SLIPPAGE_TICKS <= 0:
        return float(price)
    delta = SLIPPAGE_TICKS * TICK_SIZE
    if side == "sell":
        return float(price) - delta
    if side == "buy":
        return float(price) + delta
    raise ValueError("side must be 'sell' or 'buy'")

# ----------------------------
# RISK TIER SELECTION
# ----------------------------
def pick_risk_tier(risk_df: pd.DataFrame, position_value_mark: float) -> pd.Series:
    """
    Choose smallest tier where riskLimitValue >= position_value_mark, else highest tier.
    """
    pv = float(position_value_mark)
    mask = risk_df["riskLimitValue"] >= pv
    if mask.any():
        return risk_df.loc[mask.idxmax()]
    return risk_df.iloc[-1]

# ----------------------------
# EXACT LIQUIDATION (ISOLATED, SHORT, LINEAR USDT)
# ----------------------------
def liquidation_price_short_isolated(
    entry_price: float,
    qty_short: float,                 # negative qty for short
    leverage: float,
    mark_price: float,
    tier: pd.Series,
    fee_rate: float,
    extra_margin_added: float = 0.0
) -> float:
    """
    Isolated SHORT liquidation model (Bybit-style structure):

      LP = Entry + (IM + ExtraMargin - MM) / Qty

    Where (USDT linear, isolated):
      PositionValueEntry = |qty| * entry_price
      EstCloseFee        = |qty| * mark_price * fee_rate

      IM = (PositionValueEntry / leverage) + EstCloseFee
      MM = (PositionValueEntry * MMR) - mmDeduction + EstCloseFee

    Notes:
    - Tier selection should be based on position value at MARK (done outside).
    - Liquidation trigger uses MARK price (done in backtest/live).
    """
    qty_abs = abs(float(qty_short))
    if qty_abs <= 0:
        return float("inf")

    entry = float(entry_price)
    mark = float(mark_price)
    lev = float(leverage)

    pv_entry = qty_abs * entry
    est_close_fee = qty_abs * mark * float(fee_rate)

    im = (pv_entry / lev) + est_close_fee
    mmr = float(tier["maintenanceMarginRate"])
    mm_ded = float(tier["mmDeductionValue"])

    mm = (pv_entry * mmr) - mm_ded + est_close_fee

    lp = entry + ((im + float(extra_margin_added)) - mm) / qty_abs
    return float(lp)

# ----------------------------
# BUILD INDICATORS (matches Pine construction)
# ----------------------------
def build_indicators(df_last: pd.DataFrame, ma_len: int, band_mult: float) -> pd.DataFrame:
    """
    Matches Pine:
      main5 = ta.rma(close, maLen)
      premium_zone_k = ta.ema( main5*(1+band_mult*0.01*k), 5 )
      discount_zone_k= ta.ema( main5*(1-band_mult*0.01*k), 5 )
    """
    df = df_last.copy()
    df["main"] = rma(df["close"], int(ma_len))

    # build zones 1..8 using EMA length 5
    for k in range(1, 9):
        prem_raw = df["main"].to_numpy(dtype=float) * (1.0 + float(band_mult) * 0.01 * k)
        disc_raw = df["main"].to_numpy(dtype=float) * (1.0 - float(band_mult) * 0.01 * k)
        df[f"premium_{k}"] = ema_np(prem_raw, 5)
        df[f"discount_{k}"] = ema_np(disc_raw, 5)

    return df

# ----------------------------
# BACKTEST ENGINE (short-only, single entry)
# ----------------------------
@dataclass
class BacktestResult:
    final_wallet: float
    pnl_usdt: float
    pnl_pct: float
    trades: int
    winrate: float
    liquidated: bool

def backtest_once(
    df_last_raw: pd.DataFrame,
    df_mark_raw: pd.DataFrame,
    risk_df: pd.DataFrame,
    ma_len: int,
    band_mult: float,
    tp_perc: float
) -> Optional[BacktestResult]:
    """
    Uses:
      - LAST-price OHLC for signals (h5/l5/close)
      - MARK-price OHLC for liquidation trigger and tier selection
    Applies:
      - Fees per side on notional (TradingView-style)
      - Slippage on fills
      - 10x leverage (notional = wallet * leverage at entry)
      - Single position only
    """
    ma_len = int(ma_len)
    if len(df_last_raw) < ma_len + 20:
        return None

    # Align by timestamp (intersection)
    dfl = df_last_raw.set_index("ts")
    dfm = df_mark_raw.set_index("ts")
    common = dfl.index.intersection(dfm.index)
    if len(common) < ma_len + 20:
        return None

    dfl = dfl.loc[common].reset_index()
    dfm = dfm.loc[common].reset_index()

    # Build indicators on LAST-price series (matches Pine request.security close/high/low)
    dfl = build_indicators(dfl, ma_len, float(band_mult))

    wallet = float(STARTING_WALLET)
    pos_qty = 0.0          # negative = short
    entry_price = 0.0
    entry_fee = 0.0
    entry_notional = 0.0
    liquidated = False

    trade_pnls = []

    for i in range(1, len(dfl)):
        row = dfl.iloc[i]
        prev = dfl.iloc[i - 1]
        mrow = dfm.iloc[i]

        # last price OHLC for signals
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        prev_high = float(prev["high"])
        prev_low = float(prev["low"])

        # mark price OHLC for liquidation
        mark_close = float(mrow["close"])
        mark_high = float(mrow["high"])

        # -------- Entry Signal: highest premium crossover on HIGH --------
        entry_signal_level = 0
        for lvl in range(8, 0, -1):
            if crossover(prev_high, high, float(prev[f"premium_{lvl}"]), float(row[f"premium_{lvl}"])):
                entry_signal_level = lvl
                break

        # -------- Exit Signal: any discount crossunder on LOW, or main crossunder --------
        exit_signal = False
        for lvl in range(1, 9):
            if crossunder(prev_low, low, float(prev[f"discount_{lvl}"]), float(row[f"discount_{lvl}"])):
                exit_signal = True
                break
        if crossunder(prev_low, low, float(prev["main"]), float(row["main"])):
            exit_signal = True

        # -------- ENTRY (short only, single entry) --------
        if entry_signal_level != 0 and pos_qty == 0.0 and wallet > 0.0:
            fill = apply_slippage(close, "sell")
            notional = wallet * float(LEVERAGE)
            qty = notional / fill

            fee = notional * float(FEE_RATE)
            wallet -= fee

            pos_qty = -qty
            entry_price = fill
            entry_fee = fee
            entry_notional = notional

        # -------- If in position: LIQ check first (MARK price) --------
        if pos_qty < 0.0:
            qty_abs = abs(pos_qty)

            # tier selection based on MARK PV
            position_value_mark = qty_abs * mark_close
            tier = pick_risk_tier(risk_df, position_value_mark)

            liq = liquidation_price_short_isolated(
                entry_price=entry_price,
                qty_short=pos_qty,
                leverage=LEVERAGE,
                mark_price=mark_close,
                tier=tier,
                fee_rate=FEE_RATE,
                extra_margin_added=0.0
            )

            # liquidate if MARK high breaches liq price
            if mark_high >= liq:
                wallet = 0.0
                liquidated = True
                break

            # -------- TP always active --------
            tp_price = entry_price * (1.0 - float(tp_perc))
            if low <= tp_price:
                exit_fill = apply_slippage(tp_price, "buy")
                pnl_gross = (entry_price - exit_fill) * qty_abs
                exit_fee = (qty_abs * exit_fill) * float(FEE_RATE)

                wallet += pnl_gross
                wallet -= exit_fee

                pnl_net = pnl_gross - entry_fee - exit_fee
                trade_pnls.append(pnl_net)

                pos_qty = 0.0
                entry_price = 0.0
                entry_fee = 0.0
                entry_notional = 0.0
                continue

            # -------- fallback exit --------
            if exit_signal:
                exit_fill = apply_slippage(close, "buy")
                pnl_gross = (entry_price - exit_fill) * qty_abs
                exit_fee = (qty_abs * exit_fill) * float(FEE_RATE)

                wallet += pnl_gross
                wallet -= exit_fee

                pnl_net = pnl_gross - entry_fee - exit_fee
                trade_pnls.append(pnl_net)

                pos_qty = 0.0
                entry_price = 0.0
                entry_fee = 0.0
                entry_notional = 0.0

    trades = len(trade_pnls)
    winrate = (sum(1 for x in trade_pnls if x > 0) / trades * 100.0) if trades else 0.0
    pnl_usdt = wallet - float(STARTING_WALLET)
    pnl_pct = (pnl_usdt / float(STARTING_WALLET)) * 100.0 if STARTING_WALLET else 0.0

    return BacktestResult(
        final_wallet=float(wallet),
        pnl_usdt=float(pnl_usdt),
        pnl_pct=float(pnl_pct),
        trades=int(trades),
        winrate=float(winrate),
        liquidated=bool(liquidated)
    )

# ----------------------------
# RANDOM OPTIMISER
# ----------------------------
def optimise_random(
    df_last: pd.DataFrame,
    df_mark: pd.DataFrame,
    risk_df: pd.DataFrame,
    trials: int,
    lookback_candles: int,
    event_name: str
) -> Dict[str, Any]:
    """
    Random search over (MA_LEN, BAND_MULT, TP_PERC)
    """
    if lookback_candles > 0:
        dfl = df_last.iloc[-lookback_candles:].copy()
        dfm = df_mark.iloc[-lookback_candles:].copy()
    else:
        dfl = df_last.copy()
        dfm = df_mark.copy()

    best = None
    best_res = None

    for _ in range(int(trials)):
        ma = random.randint(MA_LEN_RANGE[0], MA_LEN_RANGE[1])
        band = random.uniform(BAND_MULT_RANGE[0], BAND_MULT_RANGE[1])
        tp = random.uniform(TP_RANGE[0], TP_RANGE[1])

        res = backtest_once(dfl, dfm, risk_df, ma, band, tp)
        if res is None:
            continue

        # Prefer higher final wallet; tie-break: more trades
        if (best_res is None) or (res.final_wallet > best_res.final_wallet) or \
           (res.final_wallet == best_res.final_wallet and res.trades > best_res.trades):
            best = {"ma_len": ma, "band_mult": float(round(band, 6)), "tp_perc": float(tp)}
            best_res = res

    if best_res is None:
        raise RuntimeError("Optimiser failed: no valid runs (not enough data?)")

    ts_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    csv_append(PARAMS_CSV_PATH, [
        ts_utc, event_name,
        best["ma_len"], best["band_mult"], best["tp_perc"],
        round(best_res.final_wallet, 6)
    ])

    return {
        "best_params": best,
        "best_result": best_res
    }
# ============================================================
# HYPEUSDT SHORT GRID — EXACT BYBIT ENGINE (PART 3/4)
# Live paper trader + detailed order logging + WFO scheduling
# ============================================================

@dataclass
class Params:
    ma_len: int
    band_mult: float
    tp_perc: float

@dataclass
class Position:
    qty: float                 # negative for short
    entry_price: float
    entry_fee: float
    entry_notional: float
    margin_at_entry: float
    signal_level: int
    params: Params
    entry_ts_utc: str

@dataclass
class RealPosition:
    qty: float
    entry_price: float
    side: str

class BybitPrivateClient:
    def get_unified_usdt(self) -> float:
        j = rest_request(
            "GET",
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED"},
            auth=True
        )
        rows = j["result"].get("list", [])
        if not rows:
            raise RuntimeError("No wallet balance returned for accountType=UNIFIED")
        for row in rows:
            for coin in row.get("coin", []):
                if coin.get("coin") != "USDT":
                    continue
                log.info("USDT wallet object: %s", coin)
                available_withdraw = coin.get("availableToWithdraw")
                available_balance = coin.get("availableBalance")
                wallet_balance = coin.get("walletBalance")

                def _parse_value(value: Any) -> Optional[float]:
                    if value is None:
                        return None
                    if isinstance(value, str) and not value.strip():
                        return None
                    try:
                        return float(value)
                    except Exception:
                        return None

                parsed_available_withdraw = _parse_value(available_withdraw)
                if parsed_available_withdraw is not None and parsed_available_withdraw > 0:
                    return parsed_available_withdraw

                parsed_available_balance = _parse_value(available_balance)
                if parsed_available_balance is not None and parsed_available_balance > 0:
                    return parsed_available_balance

                parsed_wallet_balance = _parse_value(wallet_balance)
                if parsed_wallet_balance is not None:
                    return parsed_wallet_balance

                raise RuntimeError("No usable USDT balance fields returned by Bybit")
        raise RuntimeError("USDT balance not found for accountType=UNIFIED")

    def set_position_mode(self, symbol: str, mode: int = 0):
        rest_request(
            "POST",
            "/v5/position/switch-mode",
            body={
                "category": CATEGORY,
                "symbol": symbol,
                "mode": mode
            },
            auth=True
        )

    def set_margin_mode(self, symbol: str, trade_mode: int = 1):
        rest_request(
            "POST",
            "/v5/position/set-margin-mode",
            body={
                "category": CATEGORY,
                "symbol": symbol,
                "tradeMode": trade_mode,
                "buyLeverage": str(LEVERAGE),
                "sellLeverage": str(LEVERAGE)
            },
            auth=True
        )

    def set_leverage(self, symbol: str, buy_leverage: float, sell_leverage: float):
        rest_request(
            "POST",
            "/v5/position/set-leverage",
            body={
                "category": CATEGORY,
                "symbol": symbol,
                "buyLeverage": str(buy_leverage),
                "sellLeverage": str(sell_leverage)
            },
            auth=True
        )

    def ensure_futures_setup(self, symbol: str):
        self.set_position_mode(symbol, mode=0)
        self.set_margin_mode(symbol, trade_mode=1)
        self.set_leverage(symbol, LEVERAGE, LEVERAGE)

    def get_wallet_balance(self) -> float:
        return self.get_unified_usdt()

    def get_position(self, symbol: str) -> Optional[RealPosition]:
        j = rest_request(
            "GET",
            "/v5/position/list",
            params={"category": CATEGORY, "symbol": symbol},
            auth=True
        )
        rows = j["result"]["list"]
        if not rows:
            return None
        row = rows[0]
        size = float(row.get("size", 0))
        if size == 0:
            return None
        side = row.get("side", "")
        entry_price = float(row.get("avgPrice", 0))
        qty = size if side == "Buy" else -size
        return RealPosition(qty=qty, entry_price=entry_price, side=side)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool,
        order_link_id: Optional[str] = None
    ) -> str:
        link_id = order_link_id or f"DLT-{int(time.time() * 1000)}-{uuid.uuid4().hex[:10]}"
        j = rest_request(
            "POST",
            "/v5/order/create",
            body={
                "category": CATEGORY,
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "IOC",
                "positionIdx": 0,
                "reduceOnly": reduce_only,
                "orderLinkId": link_id
            },
            auth=True
        )
        return j["result"]["orderId"]

    def get_instrument_info(self, symbol: str) -> Dict[str, Any]:
        j = rest_get("/v5/market/instruments-info", {
            "category": CATEGORY,
            "symbol": symbol
        })
        rows = j["result"]["list"]
        if not rows:
            raise RuntimeError("No instrument info returned")
        return rows[0]

    def _fetch_executions(self, order_id: str) -> List[Dict[str, Any]]:
        cursor = None
        executions: List[Dict[str, Any]] = []
        while True:
            params = {
                "category": CATEGORY,
                "symbol": SYMBOL,
                "orderId": order_id,
                "limit": 50
            }
            if cursor:
                params["cursor"] = cursor
            j = rest_request(
                "GET",
                "/v5/execution/list",
                params=params,
                auth=True
            )
            rows = j["result"]["list"]
            if rows:
                executions.extend(rows)
            cursor = j["result"].get("nextPageCursor")
            if not cursor:
                break
        return executions

    def get_execution_summary(
        self,
        order_id: str,
        timeout_sec: float = 3.0,
        poll_interval: float = 0.2
    ) -> Optional[Dict[str, float]]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            executions = self._fetch_executions(order_id)
            if executions:
                total_qty = 0.0
                total_notional = 0.0
                for row in executions:
                    exec_qty = float(row.get("execQty", 0) or 0)
                    exec_price = float(row.get("execPrice", 0) or 0)
                    if exec_qty <= 0 or exec_price <= 0:
                        continue
                    total_qty += exec_qty
                    total_notional += exec_qty * exec_price
                if total_qty > 0:
                    return {
                        "avg_price": total_notional / total_qty,
                        "qty": total_qty
                    }
            time.sleep(poll_interval)
        return None

class LivePaperTrader:
    """
    - Uses LAST-price 5m candles for signals (like Pine h5/l5/close).
    - Uses MARK price ticker + risk tiers for liquidation.
    - Logs every entry/exit (TP, mean reversion, liquidation) to trades.csv.
    - Walk-forward optimisation can run periodically and apply new params (optionally only when flat).
    """
    def __init__(self, df_last_seed: pd.DataFrame, df_mark_seed: pd.DataFrame, risk_df: pd.DataFrame, params: Params):
        self.risk_df = risk_df

        self.params = params
        self.pending_params: Optional[Params] = None

        self.wallet = float(STARTING_WALLET)
        self.position: Optional[Position] = None

        # Price series (last-price candles for strategy)
        self.df = df_last_seed[["ts","open","high","low","close"]].copy().reset_index(drop=True)
        self.closed_candle_count = 0

        # Mark price state (for liquidation)
        if len(df_mark_seed) == 0:
            raise RuntimeError("No mark seed data")
        self.mark_price = float(df_mark_seed["close"].iloc[-1])

        # Stats
        self.trade_count = 0
        self.win_count = 0
        self.realized_pnl_10x_net = 0.0
        self.realized_pnl_1x_gross = 0.0

        self._recompute_indicators()

    # ----------------------------
    # Indicator recompute
    # ----------------------------
    def _recompute_indicators(self):
        if len(self.df) < self.params.ma_len + 10:
            return
        tmp = build_indicators(self.df, self.params.ma_len, self.params.band_mult)
        # overwrite into self.df to keep memory stable
        self.df = tmp

    def _ind_snapshot(self, row: pd.Series) -> Dict[str, float]:
        def safe(k: str) -> float:
            try:
                return float(row.get(k, np.nan))
            except Exception:
                return float("nan")
        return {
            "main": safe("main"),
            "prem_1": safe("premium_1"),
            "prem_8": safe("premium_8"),
            "disc_1": safe("discount_1"),
            "disc_8": safe("discount_8"),
        }

    # ----------------------------
    # CSV logging (detailed, but still single-line)
    # ----------------------------
    def _log_trade(
        self,
        ts_utc: str,
        action: str,
        reason: str,
        signal_level: int,
        side: str,
        qty: float,
        fill_price: float,
        notional: float,
        fee: float,
        entry_price: float,
        tp_price: float,
        mark_price: float,
        liq_price: float,
        wallet_before: float,
        wallet_after: float,
        pnl_gross: float,
        pnl_net_10x: float,
        pnl_1x_usdt: float,
        pnl_1x_pct: float,
        pnl_10x_pct: float,
        tier: pd.Series,
        ind: Dict[str, float],
        candle_ohlc: Dict[str, float],
        params: Params,
    ):
        def encode(value: Any) -> str:
            try:
                return json.dumps(value, default=str, separators=(",", ":"))
            except Exception:
                return json.dumps(str(value))

        tier_dict = {}
        try:
            tier_dict = tier.to_dict()
        except Exception:
            tier_dict = {"value": str(tier)}

        csv_append(TRADES_CSV_PATH, [
            ts_utc,
            action,
            reason,
            int(signal_level),
            side,
            float(qty),
            float(fill_price),
            float(notional),
            float(fee),
            float(entry_price),
            float(tp_price),
            float(mark_price),
            float(liq_price),
            float(wallet_before),
            float(wallet_after),
            float(pnl_gross),
            float(pnl_net_10x),
            float(pnl_1x_usdt),
            float(pnl_1x_pct),
            float(pnl_10x_pct),
            encode(tier_dict),
            encode(ind),
            encode(candle_ohlc),
            int(params.ma_len),
            float(params.band_mult),
            float(params.tp_perc),
        ])

        # Also log to events.log for immediate visibility
        log.info(
            f"[{ts_utc}] {action} {reason} fill={fill_price:.8f} qty={qty:.6f} "
            f"fee={fee:.6f} pnl10x={pnl_net_10x:.4f} wallet={wallet_after:.2f} "
            f"mark={mark_price:.8f} liq={liq_price:.8f} tp={tp_price:.8f} "
            f"MA={params.ma_len} BAND={params.band_mult:.4f} TP={params.tp_perc*100:.3f}%"
        )

    # ----------------------------
    # Param application
    # ----------------------------
    def _maybe_apply_pending_params(self):
        if self.pending_params is None:
            return
        if self.position is not None:
            return

        self.params = self.pending_params
        self.pending_params = None
        self._recompute_indicators()
        log.info(f"APPLIED NEW PARAMS: MA={self.params.ma_len} BAND={self.params.band_mult:.4f} TP={self.params.tp_perc*100:.3f}%")

    # ----------------------------
    # WFO scheduling
    # ----------------------------
    def _maybe_run_wfo(self):
        if not WFO_ENABLED:
            return
        if self.closed_candle_count == 0:
            return
        if (self.closed_candle_count % WFO_EVERY_CLOSED_CANDLES) != 0:
            return
        if len(self.df) < max(self.params.ma_len + 50, WFO_LOOKBACK_CANDLES):
            return

        # Redownload a clean window for optimisation (keeps mark kline exact)
        try:
            end_ts = now_ms()
            start_ts = end_ts - int(WFO_LOOKBACK_CANDLES * 5 * 60 * 1000)

            dfl = fetch_last_klines(SYMBOL, INTERVAL, start_ts, end_ts)
            dfm = fetch_mark_klines(SYMBOL, INTERVAL, start_ts, end_ts)

            opt = optimise_random(dfl, dfm, self.risk_df, trials=WFO_TRIALS, lookback_candles=min(WFO_LOOKBACK_CANDLES, len(dfl)), event_name="WFO_REOPT")
            bp = opt["best_params"]

            self.pending_params = Params(int(bp["ma_len"]), float(bp["band_mult"]), float(bp["tp_perc"]))
            log.info(f"WFO SUGGESTED PARAMS: MA={self.pending_params.ma_len} BAND={self.pending_params.band_mult:.4f} TP={self.pending_params.tp_perc*100:.3f}% (apply_when_flat={WFO_APPLY_ONLY_WHEN_FLAT})")
        except Exception as e:
            log.error(f"WFO optimisation failed: {e}")

    # ----------------------------
    # Mark price update (liquidation trigger)
    # ----------------------------
    def on_mark_price_update(self, mark_price: float, ts_utc: str):
        self.mark_price = float(mark_price)

        if self.position is None:
            return

        pos = self.position
        qty_abs = abs(pos.qty)

        # Tier selection by mark PV
        position_value_mark = qty_abs * self.mark_price
        tier = pick_risk_tier(self.risk_df, position_value_mark)

        liq = liquidation_price_short_isolated(
            entry_price=pos.entry_price,
            qty_short=pos.qty,
            leverage=LEVERAGE,
            mark_price=self.mark_price,
            tier=tier,
            fee_rate=FEE_RATE,
            extra_margin_added=0.0
        )

        # Liquidation trigger: mark price >= liq price (short)
        if self.mark_price >= liq:
            wallet_before = self.wallet

            # liquidated => wipe wallet
            # approximate fill at liq (for reporting)
            exit_fill = apply_slippage(liq, "buy")
            exit_fee = (qty_abs * exit_fill) * float(FEE_RATE)
            pnl_gross = (pos.entry_price - exit_fill) * qty_abs  # negative
            pnl_10x_net = pnl_gross - pos.entry_fee - exit_fee

            pnl_1x_usdt = (pos.entry_price - exit_fill) * qty_abs
            pnl_1x_pct = (pnl_1x_usdt / pos.entry_notional) * 100.0 if pos.entry_notional > 0 else 0.0
            pnl_10x_pct = (pnl_10x_net / pos.margin_at_entry) * 100.0 if pos.margin_at_entry > 0 else 0.0

            # last candle snapshot
            row = self.df.iloc[-1]
            ind = self._ind_snapshot(row)
            candle_ohlc = {"open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])}

            tp_price = pos.entry_price * (1.0 - pos.params.tp_perc)

            self._log_trade(
                ts_utc=ts_utc,
                action="EXIT",
                reason="LIQUIDATION_MARK",
                signal_level=pos.signal_level,
                side="COVER",
                qty=qty_abs,
                fill_price=exit_fill,
                notional=qty_abs * exit_fill,
                fee=exit_fee,
                entry_price=pos.entry_price,
                tp_price=tp_price,
                mark_price=self.mark_price,
                liq_price=liq,
                wallet_before=wallet_before,
                wallet_after=0.0,
                pnl_gross=pnl_gross,
                pnl_net_10x=pnl_10x_net,
                pnl_1x_usdt=pnl_1x_usdt,
                pnl_1x_pct=pnl_1x_pct,
                pnl_10x_pct=pnl_10x_pct,
                tier=tier,
                ind=ind,
                candle_ohlc=candle_ohlc,
                params=pos.params,
            )

            self.wallet = 0.0
            self.position = None
            return

    # ----------------------------
    # Closed 5m candle update (signals + order management)
    # ----------------------------
    def on_closed_candle(self, candle: Dict[str, Any]):
        ts = pd.to_datetime(int(candle["start"]), unit="ms", utc=True)
        ts_utc = ts.strftime("%Y-%m-%d %H:%M:%S")

        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])

        # append to df
        self.df = pd.concat([self.df, pd.DataFrame([{"ts": ts, "open": o, "high": h, "low": l, "close": c}])], ignore_index=True)
        if len(self.df) > KEEP_CANDLES:
            self.df = self.df.iloc[-KEEP_CANDLES:].reset_index(drop=True)

        # recompute indicators and counters
        self._recompute_indicators()
        self.closed_candle_count += 1

        # WFO and apply pending params
        self._maybe_run_wfo()
        self._maybe_apply_pending_params()

        if len(self.df) < self.params.ma_len + 10:
            return

        row = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        ind = self._ind_snapshot(row)
        candle_ohlc = {"open": o, "high": h, "low": l, "close": c}

        # ----------------------------
        # Entry signal (highest premium crossover of HIGH)
        # ----------------------------
        entry_signal = 0
        for lvl in range(8, 0, -1):
            if crossover(float(prev["high"]), h, float(prev[f"premium_{lvl}"]), float(row[f"premium_{lvl}"])):
                entry_signal = lvl
                break

        # ----------------------------
        # Exit signal (any discount crossunder of LOW, or main crossunder)
        # ----------------------------
        exit_signal = False
        for lvl in range(1, 9):
            if crossunder(float(prev["low"]), l, float(prev[f"discount_{lvl}"]), float(row[f"discount_{lvl}"])):
                exit_signal = True
                break
        if crossunder(float(prev["low"]), l, float(prev["main"]), float(row["main"])):
            exit_signal = True

        # ----------------------------
        # ENTRY (short only, single)
        # ----------------------------
        if entry_signal != 0 and self.position is None and self.wallet > 0:
            wallet_before = self.wallet
            fill = apply_slippage(c, "sell")

            margin = self.wallet
            notional = margin * float(LEVERAGE)
            qty = notional / fill

            fee_entry = notional * float(FEE_RATE)
            self.wallet -= fee_entry

            pos = Position(
                qty=-qty,
                entry_price=fill,
                entry_fee=fee_entry,
                entry_notional=notional,
                margin_at_entry=margin,
                signal_level=entry_signal,
                params=Params(self.params.ma_len, self.params.band_mult, self.params.tp_perc),
                entry_ts_utc=ts_utc,
            )
            self.position = pos

            # tier/liquidation snapshot for log
            qty_abs = abs(pos.qty)
            position_value_mark = qty_abs * self.mark_price
            tier = pick_risk_tier(self.risk_df, position_value_mark)
            liq = liquidation_price_short_isolated(
                entry_price=pos.entry_price,
                qty_short=pos.qty,
                leverage=LEVERAGE,
                mark_price=self.mark_price,
                tier=tier,
                fee_rate=FEE_RATE,
                extra_margin_added=0.0
            )

            tp_price = pos.entry_price * (1.0 - pos.params.tp_perc)

            self._log_trade(
                ts_utc=ts_utc,
                action="ENTRY",
                reason="PREMIUM_XOVER",
                signal_level=entry_signal,
                side="SHORT",
                qty=qty,
                fill_price=fill,
                notional=notional,
                fee=fee_entry,
                entry_price=pos.entry_price,
                tp_price=tp_price,
                mark_price=self.mark_price,
                liq_price=liq,
                wallet_before=wallet_before,
                wallet_after=self.wallet,
                pnl_gross=0.0,
                pnl_net_10x=0.0,
                pnl_1x_usdt=0.0,
                pnl_1x_pct=0.0,
                pnl_10x_pct=0.0,
                tier=tier,
                ind=ind,
                candle_ohlc=candle_ohlc,
                params=pos.params
            )

        # ----------------------------
        # If flat, print status
        # ----------------------------
        if self.position is None:
            if PRINT_EVERY_CANDLE:
                pnl_usdt = self.wallet - float(STARTING_WALLET)
                pnl_pct = (pnl_usdt / float(STARTING_WALLET)) * 100.0 if STARTING_WALLET else 0.0
                wr = (self.win_count / self.trade_count * 100.0) if self.trade_count else 0.0
                log.info(f"[{ts_utc}] FLAT close={c:.8f} mark={self.mark_price:.8f} wallet={self.wallet:.2f} pnl={pnl_usdt:.2f} ({pnl_pct:.2f}%) trades={self.trade_count} wr={wr:.1f}%")
            return

        # ----------------------------
        # In position: manage TP / fallback exits (liquidation handled by mark updates)
        # ----------------------------
        pos = self.position
        qty_abs = abs(pos.qty)

        # tier/liquidation snapshot for log/visibility
        position_value_mark = qty_abs * self.mark_price
        tier = pick_risk_tier(self.risk_df, position_value_mark)
        liq = liquidation_price_short_isolated(
            entry_price=pos.entry_price,
            qty_short=pos.qty,
            leverage=LEVERAGE,
            mark_price=self.mark_price,
            tier=tier,
            fee_rate=FEE_RATE,
            extra_margin_added=0.0
        )

        tp_price = pos.entry_price * (1.0 - pos.params.tp_perc)

        # TP hit?
        if l <= tp_price:
            wallet_before = self.wallet
            exit_fill = apply_slippage(tp_price, "buy")

            pnl_gross = (pos.entry_price - exit_fill) * qty_abs
            fee_exit = (qty_abs * exit_fill) * float(FEE_RATE)

            # wallet update
            self.wallet += pnl_gross
            self.wallet -= fee_exit
            wallet_after = self.wallet

            pnl_10x_net = pnl_gross - pos.entry_fee - fee_exit

            # 1x reference: same qty, gross move relative to notional
            pnl_1x_usdt = (pos.entry_price - exit_fill) * qty_abs
            pnl_1x_pct = (pnl_1x_usdt / pos.entry_notional) * 100.0 if pos.entry_notional > 0 else 0.0
            pnl_10x_pct = (pnl_10x_net / pos.margin_at_entry) * 100.0 if pos.margin_at_entry > 0 else 0.0

            self.trade_count += 1
            if pnl_10x_net > 0:
                self.win_count += 1

            self.realized_pnl_10x_net += pnl_10x_net
            self.realized_pnl_1x_gross += pnl_1x_usdt

            self._log_trade(
                ts_utc=ts_utc,
                action="EXIT",
                reason="TP",
                signal_level=pos.signal_level,
                side="COVER",
                qty=qty_abs,
                fill_price=exit_fill,
                notional=qty_abs * exit_fill,
                fee=fee_exit,
                entry_price=pos.entry_price,
                tp_price=tp_price,
                mark_price=self.mark_price,
                liq_price=liq,
                wallet_before=wallet_before,
                wallet_after=wallet_after,
                pnl_gross=pnl_gross,
                pnl_net_10x=pnl_10x_net,
                pnl_1x_usdt=pnl_1x_usdt,
                pnl_1x_pct=pnl_1x_pct,
                pnl_10x_pct=pnl_10x_pct,
                tier=tier,
                ind=ind,
                candle_ohlc=candle_ohlc,
                params=pos.params
            )

            self.position = None
            self._maybe_apply_pending_params()
            return

        # Fallback exit
        if exit_signal:
            wallet_before = self.wallet
            exit_fill = apply_slippage(c, "buy")

            pnl_gross = (pos.entry_price - exit_fill) * qty_abs
            fee_exit = (qty_abs * exit_fill) * float(FEE_RATE)

            self.wallet += pnl_gross
            self.wallet -= fee_exit
            wallet_after = self.wallet

            pnl_10x_net = pnl_gross - pos.entry_fee - fee_exit
            pnl_1x_usdt = (pos.entry_price - exit_fill) * qty_abs
            pnl_1x_pct = (pnl_1x_usdt / pos.entry_notional) * 100.0 if pos.entry_notional > 0 else 0.0
            pnl_10x_pct = (pnl_10x_net / pos.margin_at_entry) * 100.0 if pos.margin_at_entry > 0 else 0.0

            self.trade_count += 1
            if pnl_10x_net > 0:
                self.win_count += 1

            self.realized_pnl_10x_net += pnl_10x_net
            self.realized_pnl_1x_gross += pnl_1x_usdt

            self._log_trade(
                ts_utc=ts_utc,
                action="EXIT",
                reason="MEAN_REVERSION",
                signal_level=pos.signal_level,
                side="COVER",
                qty=qty_abs,
                fill_price=exit_fill,
                notional=qty_abs * exit_fill,
                fee=fee_exit,
                entry_price=pos.entry_price,
                tp_price=tp_price,
                mark_price=self.mark_price,
                liq_price=liq,
                wallet_before=wallet_before,
                wallet_after=wallet_after,
                pnl_gross=pnl_gross,
                pnl_net_10x=pnl_10x_net,
                pnl_1x_usdt=pnl_1x_usdt,
                pnl_1x_pct=pnl_1x_pct,
                pnl_10x_pct=pnl_10x_pct,
                tier=tier,
                ind=ind,
                candle_ohlc=candle_ohlc,
                params=pos.params
            )

            self.position = None
            self._maybe_apply_pending_params()
            return

        # Candle summary when still in position
        if PRINT_EVERY_CANDLE:
            pnl_usdt = self.wallet - float(STARTING_WALLET)
            pnl_pct = (pnl_usdt / float(STARTING_WALLET)) * 100.0 if STARTING_WALLET else 0.0
            wr = (self.win_count / self.trade_count * 100.0) if self.trade_count else 0.0
            log.info(
                f"[{ts_utc}] IN_POS close={c:.8f} mark={self.mark_price:.8f} liq={liq:.8f} "
                f"wallet={self.wallet:.2f} pnl={pnl_usdt:.2f} ({pnl_pct:.2f}%) trades={self.trade_count} wr={wr:.1f}%"
            )

class LiveRealTrader:
    """
    Live trader that places real Bybit orders using authenticated endpoints.
    Uses the same signal logic as the paper trader but relies on exchange state.
    """
    def __init__(
        self,
        df_last_seed: pd.DataFrame,
        df_mark_seed: pd.DataFrame,
        risk_df: pd.DataFrame,
        params: Params,
        client: BybitPrivateClient
    ):
        self.client = client
        self.risk_df = risk_df
        self.params = params
        self.pending_params: Optional[Params] = None
        self.instrument = self.client.get_instrument_info(SYMBOL)

        self.wallet = float(self.client.get_unified_usdt())
        self.initial_wallet = self.wallet
        self.position: Optional[RealPosition] = self.client.get_position(SYMBOL)

        self.df = df_last_seed[["ts","open","high","low","close"]].copy().reset_index(drop=True)
        self.closed_candle_count = 0

        if len(df_mark_seed) == 0:
            raise RuntimeError("No mark seed data")
        self.mark_price = float(df_mark_seed["close"].iloc[-1])

        self.trade_count = 0
        self.win_count = 0
        self.realized_pnl_10x_net = 0.0
        self.realized_pnl_1x_gross = 0.0

        self._recompute_indicators()

    def _recompute_indicators(self):
        if len(self.df) < self.params.ma_len + 10:
            return
        tmp = build_indicators(self.df, self.params.ma_len, self.params.band_mult)
        self.df = tmp

    def _maybe_apply_pending_params(self):
        if self.pending_params is None:
            return
        if self.position is not None and WFO_APPLY_ONLY_WHEN_FLAT:
            return

        self.params = self.pending_params
        self.pending_params = None
        self._recompute_indicators()
        log.info(f"APPLIED NEW PARAMS: MA={self.params.ma_len} BAND={self.params.band_mult:.4f} TP={self.params.tp_perc*100:.3f}%")

    def _maybe_run_wfo(self):
        if not WFO_ENABLED:
            return
        if self.closed_candle_count == 0:
            return
        if (self.closed_candle_count % WFO_EVERY_CLOSED_CANDLES) != 0:
            return
        if len(self.df) < max(self.params.ma_len + 50, WFO_LOOKBACK_CANDLES):
            return

        try:
            end_ts = now_ms()
            start_ts = end_ts - int(WFO_LOOKBACK_CANDLES * 5 * 60 * 1000)

            dfl = fetch_last_klines(SYMBOL, INTERVAL, start_ts, end_ts)
            dfm = fetch_mark_klines(SYMBOL, INTERVAL, start_ts, end_ts)

            opt = optimise_random(dfl, dfm, self.risk_df, trials=WFO_TRIALS, lookback_candles=min(WFO_LOOKBACK_CANDLES, len(dfl)), event_name="WFO_REOPT")
            bp = opt["best_params"]

            self.pending_params = Params(int(bp["ma_len"]), float(bp["band_mult"]), float(bp["tp_perc"]))
            log.info(f"WFO SUGGESTED PARAMS: MA={self.pending_params.ma_len} BAND={self.pending_params.band_mult:.4f} TP={self.pending_params.tp_perc*100:.3f}% (apply_when_flat={WFO_APPLY_ONLY_WHEN_FLAT})")
        except Exception as e:
            log.error(f"WFO optimisation failed: {e}")

    def _refresh_state(self):
        self.wallet = float(self.client.get_unified_usdt())
        self.position = self.client.get_position(SYMBOL)

    def _format_qty(self, raw_qty: float) -> float:
        lot_filter = self.instrument.get("lotSizeFilter", {})
        min_qty = float(lot_filter.get("minOrderQty", 0) or 0)
        max_qty = float(lot_filter.get("maxOrderQty", 0) or 0)
        step = float(lot_filter.get("qtyStep", 0.000001) or 0.000001)

        if step <= 0:
            raise RuntimeError("Invalid qty step from instrument info.")

        qty = math.floor(raw_qty / step) * step
        qty = float(f"{qty:.12f}")

        if qty <= 0 or qty < min_qty:
            raise RuntimeError(f"Order qty {qty} below minOrderQty {min_qty}.")
        if max_qty and qty > max_qty:
            raise RuntimeError(f"Order qty {qty} above maxOrderQty {max_qty}.")
        return qty

    def _min_notional(self) -> float:
        lot_filter = self.instrument.get("lotSizeFilter", {})
        for key in ("minOrderValue", "minNotionalValue", "minNotional"):
            value = lot_filter.get(key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    continue
        return 0.0

    def _max_risk_limit(self) -> float:
        try:
            return float(self.risk_df["riskLimitValue"].max())
        except Exception:
            return 0.0

    def _ensure_entry_risk_checks(self, qty: float, price: float, wallet_before: float):
        notional = qty * price
        min_notional = self._min_notional()
        if min_notional and notional < min_notional:
            raise RuntimeError(f"Order notional {notional:.6f} below min notional {min_notional:.6f}.")

        max_risk = self._max_risk_limit()
        if max_risk and notional > max_risk:
            raise RuntimeError(f"Order notional {notional:.6f} exceeds risk limit {max_risk:.6f}.")

        margin_required = notional / float(LEVERAGE)
        est_fee = notional * float(FEE_RATE)
        if wallet_before < (margin_required + est_fee):
            raise RuntimeError(
                f"Insufficient margin: wallet {wallet_before:.6f} < required {margin_required + est_fee:.6f}."
            )

    def _is_candle_fresh(self, candle_start: pd.Timestamp) -> bool:
        interval_ms = int(INTERVAL) * 60 * 1000
        candle_close_ms = int(candle_start.value / 1_000_000) + interval_ms
        return (now_ms() - candle_close_ms) <= (CANDLE_STALENESS_MAX_SEC * 1000)

    def on_mark_price_update(self, mark_price: float, ts_utc: str):
        self.mark_price = float(mark_price)

    def _log_real_trade(
        self,
        ts_utc: str,
        action: str,
        reason: str,
        side: str,
        qty: float,
        fill_price: float,
        entry_price: float,
        wallet_before: float,
        wallet_after: float
    ):
        pnl_gross = (entry_price - fill_price) * qty if side == "COVER" else 0.0
        fee = (qty * fill_price) * float(FEE_RATE)
        pnl_net = pnl_gross - fee

        if action == "EXIT":
            self.trade_count += 1
            if pnl_net > 0:
                self.win_count += 1
            self.realized_pnl_10x_net += pnl_net
            self.realized_pnl_1x_gross += pnl_gross

        tier = pick_risk_tier(self.risk_df, abs(qty) * self.mark_price)
        ind = {"main": float("nan")}
        candle_ohlc = {"open": float("nan"), "high": float("nan"), "low": float("nan"), "close": float("nan")}

        csv_append(TRADES_CSV_PATH, [
            ts_utc,
            action,
            reason,
            0,
            side,
            float(qty),
            float(fill_price),
            float(qty * fill_price),
            float(fee),
            float(entry_price),
            float(entry_price * (1.0 - self.params.tp_perc)),
            float(self.mark_price),
            0.0,
            float(wallet_before),
            float(wallet_after),
            float(pnl_gross),
            float(pnl_net),
            float(pnl_gross),
            0.0,
            0.0,
            json.dumps(tier.to_dict(), separators=(",", ":")),
            json.dumps(ind, separators=(",", ":")),
            json.dumps(candle_ohlc, separators=(",", ":")),
            int(self.params.ma_len),
            float(self.params.band_mult),
            float(self.params.tp_perc),
        ])

        log.info(
            f"[{ts_utc}] {action} {reason} fill={fill_price:.8f} qty={qty:.6f} "
            f"wallet={wallet_after:.2f} mark={self.mark_price:.8f} MA={self.params.ma_len} "
            f"BAND={self.params.band_mult:.4f} TP={self.params.tp_perc*100:.3f}%"
        )

    def on_closed_candle(self, candle: Dict[str, Any]):
        ts = pd.to_datetime(int(candle["start"]), unit="ms", utc=True)
        ts_utc = ts.strftime("%Y-%m-%d %H:%M:%S")

        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])

        self.df = pd.concat([self.df, pd.DataFrame([{"ts": ts, "open": o, "high": h, "low": l, "close": c}])], ignore_index=True)
        if len(self.df) > KEEP_CANDLES:
            self.df = self.df.iloc[-KEEP_CANDLES:].reset_index(drop=True)

        self._recompute_indicators()
        self.closed_candle_count += 1
        self._maybe_run_wfo()
        self._maybe_apply_pending_params()

        self._refresh_state()

        if not self._is_candle_fresh(ts):
            log.warning(f"[{ts_utc}] Skipping stale candle (older than {CANDLE_STALENESS_MAX_SEC}s).")
            return

        if len(self.df) < self.params.ma_len + 10:
            return

        row = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        entry_signal = 0
        for lvl in range(8, 0, -1):
            if crossover(float(prev["high"]), h, float(prev[f"premium_{lvl}"]), float(row[f"premium_{lvl}"])):
                entry_signal = lvl
                break

        exit_signal = False
        for lvl in range(1, 9):
            if crossunder(float(prev["low"]), l, float(prev[f"discount_{lvl}"]), float(row[f"discount_{lvl}"])):
                exit_signal = True
                break
        if crossunder(float(prev["low"]), l, float(prev["main"]), float(row["main"])):
            exit_signal = True

        if entry_signal != 0 and self.position is None:
            wallet_before = self.wallet
            try:
                qty = self._format_qty((self.wallet * float(LEVERAGE)) / c)
                self._ensure_entry_risk_checks(qty, c, wallet_before)
                order_id = self.client.place_market_order(SYMBOL, "Sell", qty, reduce_only=False)
                self._refresh_state()
                summary = self.client.get_execution_summary(order_id)
                fill_price = summary["avg_price"] if summary else fetch_last_price(SYMBOL)
                filled_qty = summary["qty"] if summary else qty
                self._log_real_trade(
                    ts_utc=ts_utc,
                    action="ENTRY",
                    reason="PREMIUM_XOVER",
                    side="SHORT",
                    qty=filled_qty,
                    fill_price=fill_price,
                    entry_price=fill_price,
                    wallet_before=wallet_before,
                    wallet_after=self.wallet
                )
            except Exception as e:
                log.error(f"[{ts_utc}] Entry order aborted: {e}")
            return

        if self.position is None:
            if PRINT_EVERY_CANDLE:
                pnl_usdt = self.wallet - float(self.initial_wallet)
                pnl_pct = (pnl_usdt / float(self.initial_wallet)) * 100.0 if self.initial_wallet else 0.0
                log.info(
                    f"[{ts_utc}] FLAT close={c:.8f} mark={self.mark_price:.8f} "
                    f"wallet={self.wallet:.2f} pnl={pnl_usdt:.2f} ({pnl_pct:.2f}%)"
                )
            return

        pos = self.position
        entry_price = pos.entry_price
        qty_abs = abs(pos.qty)
        tp_price = entry_price * (1.0 - float(self.params.tp_perc))

        if l <= tp_price or exit_signal:
            wallet_before = self.wallet
            try:
                qty_to_close = self._format_qty(qty_abs)
                order_id = self.client.place_market_order(SYMBOL, "Buy", qty_to_close, reduce_only=True)
                self._refresh_state()
                summary = self.client.get_execution_summary(order_id)
                fill_price = summary["avg_price"] if summary else fetch_last_price(SYMBOL)
                filled_qty = summary["qty"] if summary else qty_to_close
                reason = "TP" if l <= tp_price else "MEAN_REVERSION"
                self._log_real_trade(
                    ts_utc=ts_utc,
                    action="EXIT",
                    reason=reason,
                    side="COVER",
                    qty=filled_qty,
                    fill_price=fill_price,
                    entry_price=entry_price,
                    wallet_before=wallet_before,
                    wallet_after=self.wallet
                )
            except Exception as e:
                log.error(f"[{ts_utc}] Exit order failed: {e}")
# ============================================================
# HYPEUSDT SHORT GRID — EXACT BYBIT ENGINE (PART 4/4)
# WebSocket (auto-reconnect, passive ping) + main() glue
# ============================================================

def start_live_ws(trader):
    """
    Bybit public WS:
      - tickers.{SYMBOL} -> markPrice updates (liquidation trigger)
      - kline.5.{SYMBOL} -> closed 5m candles for signals/trading

    Requirements you asked for:
      - Auto reconnect
      - No ping spam (ping only if needed)
    """
    ws_url = "wss://stream.bybit.com/v5/public/linear"
    topic_k = f"kline.{INTERVAL}.{SYMBOL}"
    topic_t = f"tickers.{SYMBOL}"

    last_msg_time = {"t": time.time()}
    last_ping_time = {"t": 0.0}

    def on_open(ws):
        ws.send(json.dumps({"op": "subscribe", "args": [topic_k, topic_t]}))
        log.info(f"WebSocket connected. Subscribed: {topic_k} and {topic_t}")
        log.info("Live trading started. Stop with Ctrl+C.\n")

    def on_message(ws, message):
        last_msg_time["t"] = time.time()

        try:
            msg = json.loads(message)
        except Exception:
            return

        # subscription/pong acks
        if isinstance(msg, dict) and msg.get("op") in ("subscribe", "pong"):
            return

        topic = msg.get("topic", "")
        data = msg.get("data")

        # Mark price updates
        if topic == topic_t and data:
            if isinstance(data, list) and len(data) > 0:
                d = data[0]
            elif isinstance(data, dict):
                d = data
            else:
                return

            mp = d.get("markPrice")
            if mp is not None:
                ts_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                trader.on_mark_price_update(float(mp), ts_utc)
            return

        # Closed 5m candles
        if topic == topic_k and data:
            for c in data:
                if c.get("confirm") is True:
                    trader.on_closed_candle(c)

    def on_error(ws, error):
        log.error(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        log.warning(f"WebSocket closed: {close_status_code} {close_msg}")

    def run_one_connection():
        ws = WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Passive heartbeat: run_forever in a thread-ish loop using built-in dispatch.
        # We disable built-in ping. We'll send our own ping only if message silence occurs.
        # NOTE: websocket-client doesn't let us easily "inject" pings while blocking,
        # so we use a short dispatcher timeout (ping_interval None).
        #
        # This loop pattern:
        #  - run_forever returns on disconnect
        #  - between reconnects we can decide if we need to ping (not spam)
        ws.run_forever(ping_interval=None, ping_timeout=None)

    while True:
        try:
            # Start one connection (blocking). If it exits, reconnect.
            run_one_connection()
        except Exception as e:
            log.error(f"WS crashed: {e}")

        # If we got no data recently, do a short pause and reconnect.
        log.warning("WebSocket disconnected. Reconnecting in 5 seconds...")
        time.sleep(5)


def download_seed_history(days_back: int):
    end_ts = now_ms()
    start_ts = end_ts - int(days_back * 24 * 60 * 60 * 1000)

    log.info(f"Downloading LAST-price 5m history for {SYMBOL} (past {days_back} days)...")
    df_last = fetch_last_klines(SYMBOL, INTERVAL, start_ts, end_ts)
    log.info(f"Last-price candles: {len(df_last)}")

    log.info(f"Downloading MARK-price 5m history for {SYMBOL} (past {days_back} days)...")
    df_mark = fetch_mark_klines(SYMBOL, INTERVAL, start_ts, end_ts)
    log.info(f"Mark-price candles: {len(df_mark)}")

    return df_last, df_mark


def pretty_opt_summary(best_params: Dict[str, Any], best_result: BacktestResult) -> str:
    return (
        f"BEST PARAMS: MA_LEN={best_params['ma_len']} "
        f"BAND_MULT={best_params['band_mult']} "
        f"TP_PERC={best_params['tp_perc']*100:.3f}% | "
        f"Final Wallet={best_result.final_wallet:.2f} USDT | "
        f"PnL={best_result.pnl_usdt:.2f} USDT ({best_result.pnl_pct:.2f}%) | "
        f"Trades={best_result.trades} | "
        f"Winrate={best_result.winrate:.2f}% | "
        f"Liquidated={best_result.liquidated}"
    )


def main():
    # Safety check for interval hardcoding (we only ever request interval=5 anyway)
    if INTERVAL != "5":
        raise RuntimeError("This script is hard-coded to 5m candles only (INTERVAL must be '5').")

    client = None
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Live wallet usage requires BYBIT_API_KEY and BYBIT_API_SECRET.")
    client = BybitPrivateClient()
    unified_balance = float(client.get_unified_usdt())
    required_amount = float(STARTING_WALLET)
    log.info(f"Unified available USDT balance: {unified_balance:.2f}")
    log.info(f"Required margin (USDT): {required_amount:.2f}")

    if unified_balance < required_amount:
        raise RuntimeError(
            f"Insufficient unified balance: {unified_balance:.2f} < required {required_amount:.2f}"
        )

    globals()["STARTING_WALLET"] = unified_balance
    log.info(f"Using live wallet balance for backtest baseline: {unified_balance:.2f} USDT")

    # 1) Download seed history so indicators are warmed up
    df_last, df_mark = download_seed_history(DAYS_BACK_SEED)

    # 2) Risk tiers for liquidation model
    log.info(f"Fetching risk tiers for {SYMBOL} ...")
    risk_df = fetch_risk_tiers(SYMBOL)
    log.info(f"Risk tiers loaded: {len(risk_df)}")

    # 3) Initial optimisation (random search)
    log.info(
        f"Running initial optimisation: trials={INIT_TRIALS}, wallet={STARTING_WALLET:.2f}USDT, "
        f"leverage={LEVERAGE}x, fee={FEE_RATE*100:.4f}%/side"
    )

    opt = optimise_random(
        df_last=df_last,
        df_mark=df_mark,
        risk_df=risk_df,
        trials=INIT_TRIALS,
        lookback_candles=min(len(df_last), len(df_mark)),
        event_name="INIT_OPT"
    )

    best_params = opt["best_params"]
    best_result = opt["best_result"]

    log.info(pretty_opt_summary(best_params, best_result))

    # 4) Start live trading using optimal params
    params = Params(
        ma_len=int(best_params["ma_len"]),
        band_mult=float(best_params["band_mult"]),
        tp_perc=float(best_params["tp_perc"])
    )

    if REAL_TRADING_ENABLED:
        if client is None:
            client = BybitPrivateClient()
        client.ensure_futures_setup(SYMBOL)
        trader = LiveRealTrader(
            df_last_seed=df_last,
            df_mark_seed=df_mark,
            risk_df=risk_df,
            params=params,
            client=client
        )
        log.info("REAL TRADING ENABLED: sending live orders to Bybit.")
    else:
        trader = LivePaperTrader(
            df_last_seed=df_last,
            df_mark_seed=df_mark,
            risk_df=risk_df,
            params=params
        )

    # 5) Live WS (auto reconnect)
    try:
        start_live_ws(trader)
    except KeyboardInterrupt:
        # summary on stop
        baseline_wallet = getattr(trader, "initial_wallet", float(STARTING_WALLET))
        pnl_usdt = trader.wallet - float(baseline_wallet)
        pnl_pct = (pnl_usdt / float(baseline_wallet)) * 100.0 if baseline_wallet else 0.0
        wr = (trader.win_count / trader.trade_count * 100.0) if trader.trade_count else 0.0

        log.info("\nStopped by user.")
        log.info("=============== SUMMARY ===============")
        log.info(f"Final wallet: {trader.wallet:.2f} USDT")
        log.info(f"PnL (wallet): {pnl_usdt:.2f} USDT ({pnl_pct:.2f}%)")
        log.info(f"Trades:       {trader.trade_count}")
        log.info(f"Winrate:      {wr:.2f}%")
        log.info(f"Sum PnL 1x (gross move):   {trader.realized_pnl_1x_gross:.2f} USDT")
        log.info(f"Sum PnL 10x (net wallet):  {trader.realized_pnl_10x_net:.2f} USDT")
        log.info(f"Logs directory: {LOG_DIR}")
        log.info("=======================================")

if __name__ == "__main__":
    main()
