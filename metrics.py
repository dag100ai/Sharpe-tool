"""
Risk and return metrics, computed from daily prices.

Every function here is pure: it takes pandas Series of prices or daily
returns and gives back numbers. Nothing in this file fetches data or
touches the user interface, so the maths can be tested on its own.

Conventions
-----------
* Daily returns are simple returns: r_t = P_t / P_(t-1) - 1.
* Prices are adjusted for splits, bonuses and dividends, so returns are
  total returns.
* Annualisation uses 252 trading days (the global convention).
* The annual risk-free rate is converted to a daily rate by compounding:
  (1 + rf) ** (1/252) - 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Below this many daily returns the numbers are shown with a warning.
LOW_CONFIDENCE_DAYS = 60
# Below this many daily returns nothing is computed at all.
MINIMUM_DAYS = 20
# Share of no-trade / unchanged-price days that triggers a liquidity warning.
STALE_DAY_THRESHOLD = 0.10


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def daily_rf(annual_rf: float) -> float:
    """Convert an annual risk-free rate (0.055 = 5.5%) to a daily rate."""
    return (1.0 + annual_rf) ** (1.0 / TRADING_DAYS) - 1.0


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns from a price series."""
    return prices.pct_change().dropna()


def _safe_div(a: float, b: float) -> float:
    if b is None or a is None or b == 0 or np.isnan(b) or np.isnan(a):
        return float("nan")
    return a / b


# --------------------------------------------------------------------------
# Stand-alone metrics (one price series)
# --------------------------------------------------------------------------

def sharpe_ratio(r: pd.Series, annual_rf: float) -> float:
    """Annualised Sharpe ratio: mean daily excess return / stdev of daily
    returns, scaled by sqrt(252)."""
    excess = r - daily_rf(annual_rf)
    sd = r.std(ddof=1)
    return _safe_div(excess.mean(), sd) * math.sqrt(TRADING_DAYS)


def sortino_ratio(r: pd.Series, annual_rf: float) -> float:
    """Annualised Sortino ratio. The denominator is downside deviation:
    only days that returned less than the risk-free rate count as risk."""
    excess = r - daily_rf(annual_rf)
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    return _safe_div(excess.mean(), downside) * math.sqrt(TRADING_DAYS)


def annual_volatility(r: pd.Series) -> float:
    return r.std(ddof=1) * math.sqrt(TRADING_DAYS)


def total_return(prices: pd.Series) -> float:
    return prices.iloc[-1] / prices.iloc[0] - 1.0


def cagr(prices: pd.Series) -> float:
    """Compound annual growth rate over calendar time."""
    days = (prices.index[-1] - prices.index[0]).days
    if days <= 0:
        return float("nan")
    return (prices.iloc[-1] / prices.iloc[0]) ** (365.25 / days) - 1.0


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Percentage below the running peak on each day (0 = at a new high)."""
    return prices / prices.cummax() - 1.0


@dataclass
class DrawdownInfo:
    max_drawdown: float          # negative number, e.g. -0.23
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None   # None = not yet recovered
    days_to_recover: int | None          # calendar days, trough to recovery
    current_drawdown: float              # where the stock stands today


def drawdown_info(prices: pd.Series) -> DrawdownInfo:
    dd = drawdown_series(prices)
    trough = dd.idxmin()
    peak = prices.loc[:trough].idxmax()
    peak_value = prices.loc[peak]
    after = prices.loc[trough:]
    recovered = after[after >= peak_value]
    rec_date = recovered.index[0] if len(recovered) and dd.min() < 0 else None
    return DrawdownInfo(
        max_drawdown=float(dd.min()),
        peak_date=peak,
        trough_date=trough,
        recovery_date=rec_date,
        days_to_recover=(rec_date - trough).days if rec_date is not None else None,
        current_drawdown=float(dd.iloc[-1]),
    )


def calmar_ratio(prices: pd.Series) -> float:
    mdd = drawdown_series(prices).min()
    if mdd >= 0:
        return float("nan")
    return cagr(prices) / abs(mdd)


def value_at_risk(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical one-day VaR and expected shortfall (CVaR), both returned
    as positive loss fractions. VaR 0.03 = on the worst 5% of days the
    stock lost more than 3%."""
    cutoff = np.percentile(r, (1 - level) * 100)
    tail = r[r <= cutoff]
    return float(-cutoff), float(-tail.mean())


def stale_day_share(prices: pd.Series, volume: pd.Series | None = None) -> float:
    """Share of trading days with no trades, or an unchanged closing price.
    A high share means thin trading, which understates volatility."""
    unchanged = (prices.diff().iloc[1:] == 0)
    if volume is not None and len(volume) == len(prices):
        no_trade = (volume.iloc[1:] == 0)
        unchanged = unchanged | no_trade
    return float(unchanged.mean()) if len(unchanged) else float("nan")


def rolling_sharpe(r: pd.Series, annual_rf: float, window: int) -> pd.Series:
    excess = r - daily_rf(annual_rf)
    return (excess.rolling(window).mean() / r.rolling(window).std(ddof=1)) * math.sqrt(TRADING_DAYS)


def rolling_window_for(n_returns: int) -> int | None:
    """Pick a rolling window that leaves enough points to draw a line."""
    if n_returns >= 180:
        return 90
    if n_returns >= 126:
        return 63
    return None


# --------------------------------------------------------------------------
# Relative metrics (stock against a benchmark or another stock)
# --------------------------------------------------------------------------

@dataclass
class RelativeMetrics:
    beta: float
    alpha: float               # annualised Jensen's alpha
    correlation: float
    upside_capture: float      # 1.0 = captured 100% of benchmark up days
    downside_capture: float    # 1.0 = took 100% of benchmark down days
    tracking_error: float      # annualised
    information_ratio: float
    common_days: int


def align(*series: pd.Series) -> pd.DataFrame:
    """Keep only the dates every series traded on."""
    return pd.concat(series, axis=1, join="inner").dropna()


def relative_metrics(r_stock: pd.Series, r_bench: pd.Series, annual_rf: float) -> RelativeMetrics:
    df = align(r_stock.rename("s"), r_bench.rename("b"))
    s, b = df["s"], df["b"]
    rf = daily_rf(annual_rf)

    beta = _safe_div(np.cov(s, b, ddof=1)[0, 1], b.var(ddof=1))
    alpha = ((s - rf).mean() - beta * (b - rf).mean()) * TRADING_DAYS

    up, down = b > 0, b < 0
    up_cap = _safe_div(s[up].mean(), b[up].mean()) if up.any() else float("nan")
    down_cap = _safe_div(s[down].mean(), b[down].mean()) if down.any() else float("nan")

    active = s - b
    te = active.std(ddof=1) * math.sqrt(TRADING_DAYS)
    ir = _safe_div(active.mean() * TRADING_DAYS, te)

    return RelativeMetrics(
        beta=float(beta),
        alpha=float(alpha),
        correlation=float(s.corr(b)),
        upside_capture=float(up_cap),
        downside_capture=float(down_cap),
        tracking_error=float(te),
        information_ratio=float(ir),
        common_days=len(df),
    )


# --------------------------------------------------------------------------
# One call that computes everything for a price series
# --------------------------------------------------------------------------

@dataclass
class Summary:
    start: pd.Timestamp
    end: pd.Timestamp
    n_returns: int
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    var95: float
    cvar95: float
    drawdown: DrawdownInfo
    stale_share: float
    low_confidence: bool


def summarise(prices: pd.Series, annual_rf: float, volume: pd.Series | None = None) -> Summary:
    prices = prices.dropna()
    r = daily_returns(prices)
    if len(r) < MINIMUM_DAYS:
        raise ValueError(
            f"Only {len(r)} trading days of returns. At least {MINIMUM_DAYS} are "
            "needed for any meaningful figure. Pick an earlier buy date."
        )
    var95, cvar95 = value_at_risk(r)
    return Summary(
        start=prices.index[0],
        end=prices.index[-1],
        n_returns=len(r),
        total_return=total_return(prices),
        cagr=cagr(prices),
        volatility=annual_volatility(r),
        sharpe=sharpe_ratio(r, annual_rf),
        sortino=sortino_ratio(r, annual_rf),
        calmar=calmar_ratio(prices),
        var95=var95,
        cvar95=cvar95,
        drawdown=drawdown_info(prices),
        stale_share=stale_day_share(prices, volume),
        low_confidence=len(r) < LOW_CONFIDENCE_DAYS,
    )
