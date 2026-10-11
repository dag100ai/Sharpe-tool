"""
Market data layer.

This is the only file that knows where prices come from. The rest of the
app calls `get_source()` and uses the three methods defined on
`PriceSource`. To switch providers (for example to nsepython for a
private, locally-run copy), write a new class with the same methods and
return it from `get_source()`. Nothing else needs to change.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}

# Fewer rows than this means the provider has no usable history for the
# symbol (Yahoo sometimes returns a single latest quote instead of history).
USABLE_ROWS = 5


class DataError(Exception):
    """Raised with a message that can be shown to the user as-is."""


@dataclass
class PriceData:
    symbol: str                 # the provider symbol actually used, e.g. INFY.NS
    close: pd.Series            # adjusted close (splits, bonuses, dividends)
    volume: pd.Series | None
    note: str | None = None     # anything the user should know about the data


@dataclass
class Profile:
    name: str
    sector: str | None
    industry: str | None


class PriceSource(Protocol):
    name: str

    def history(self, symbol: str, start: dt.date, end: dt.date) -> PriceData: ...
    def profile(self, symbol: str) -> Profile: ...
    def search(self, query: str) -> list[tuple[str, str]]: ...


# --------------------------------------------------------------------------
# Yahoo Finance implementation
# --------------------------------------------------------------------------

class YahooSource:
    name = "Yahoo Finance"

    def history(self, symbol: str, start: dt.date, end: dt.date) -> PriceData:
        import yfinance as yf

        try:
            df = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),
                auto_adjust=True,      # adjusts for splits, bonuses and dividends
                actions=False,
            )
        except Exception as exc:  # network errors, rate limits
            raise DataError(
                f"Could not reach {self.name} for {symbol}. Try again in a minute. ({exc})"
            ) from exc

        if df is None or df.empty or "Close" not in df:
            return PriceData(symbol, pd.Series(dtype=float), None)

        # Yahoo stamps Indian prices with the exchange time zone. Strip it and
        # keep only the date so stocks and indices line up day by day.
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()

        close = df["Close"].astype(float)
        close = close[close > 0].dropna()
        volume = df["Volume"].reindex(close.index) if "Volume" in df else None
        return PriceData(symbol, close, volume)

    def profile(self, symbol: str) -> Profile:
        import yfinance as yf

        try:
            info = yf.Ticker(symbol).info or {}
        except Exception:
            info = {}
        return Profile(
            name=info.get("longName") or info.get("shortName") or symbol,
            sector=info.get("sector"),
            industry=info.get("industry"),
        )

    def search(self, query: str) -> list[tuple[str, str]]:
        """Indian listings matching a company name: [(symbol, name), ...]."""
        import yfinance as yf

        try:
            quotes = yf.Search(query, max_results=10).quotes
        except Exception:
            return []
        out = []
        for q in quotes:
            sym = q.get("symbol", "")
            if sym.endswith((".NS", ".BO")) and q.get("quoteType") == "EQUITY":
                out.append((sym, q.get("shortname") or q.get("longname") or sym))
        return out


def get_source() -> PriceSource:
    """Change this one line to swap data providers."""
    return YahooSource()


# --------------------------------------------------------------------------
# Ticker handling
# --------------------------------------------------------------------------

def clean_ticker(raw: str) -> str:
    return re.sub(r"\s+", "", raw or "").upper()


def candidate_symbols(raw: str, exchange: str) -> list[str]:
    """The symbol to try first, then the same company on the other exchange.

    'INFY' on BSE -> ['INFY.BO', 'INFY.NS']. A user may also type the full
    provider symbol ('INFY.BO'); that is respected.
    """
    t = clean_ticker(raw)
    if not t:
        raise DataError("Enter a ticker symbol, for example RELIANCE or TCS.")
    if t.isdigit():
        raise DataError(
            f"'{t}' looks like a BSE scrip code. Enter the trading symbol instead "
            "(for example RELIANCE rather than 500325), or search by company name."
        )
    if t.endswith((".NS", ".BO")):
        base, own = t[:-3], t[-3:]
    else:
        base, own = t, EXCHANGE_SUFFIX[exchange]
    other = ".BO" if own == ".NS" else ".NS"
    return [base + own, base + other]


def load_prices(raw: str, exchange: str, start: dt.date, end: dt.date,
                source: PriceSource | None = None) -> PriceData:
    """Fetch history for the requested listing, falling back to the same
    company on the other exchange if the provider has no history for it.
    The fallback is always disclosed in `note`."""
    source = source or get_source()
    first, second = candidate_symbols(raw, exchange)

    data = source.history(first, start, end)
    if len(data.close) >= USABLE_ROWS:
        return data

    alt = source.history(second, start, end)
    if len(alt.close) >= USABLE_ROWS:
        ex_first = "NSE" if first.endswith(".NS") else "BSE"
        ex_alt = "NSE" if second.endswith(".NS") else "BSE"
        alt.note = (
            f"{source.name} has no {ex_first} price history for {first[:-3]}, so "
            f"{ex_alt} prices for the same company are used. Prices on the two "
            "exchanges track each other closely, so the figures are effectively the same."
        )
        return alt

    suggestions = source.search(first[:-3])
    hint = ""
    if suggestions:
        listed = ", ".join(f"{s} ({n})" for s, n in suggestions[:4])
        hint = f" Did you mean: {listed}?"
    raise DataError(
        f"No price history found for {first[:-3]} on NSE or BSE.{hint}"
    )
