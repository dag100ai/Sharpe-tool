"""
Plain-language reading of the metrics.

Fixed rules, no AI. The wording describes what happened over the period;
it never says whether to buy, sell or hold.
"""

from __future__ import annotations

import math

from metrics import STALE_DAY_THRESHOLD, RelativeMetrics, Summary


def _ok(x: float | None) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x: float | None, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%" if _ok(x) else "n/a"


def num(x: float | None, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if _ok(x) else "n/a"


def day(ts) -> str:
    return ts.strftime("%d %b %Y")


def sharpe_band(s: float) -> str:
    if not _ok(s):
        return "Not measurable"
    if s < 0:
        return "Below the risk-free rate"
    if s < 0.5:
        return "Weak"
    if s < 1.0:
        return "Moderate"
    if s < 2.0:
        return "Good"
    return "Very strong"


def sharpe_sentence(s: float) -> str:
    if not _ok(s):
        return "Prices did not move enough over this period to measure risk."
    if s < 0:
        return ("The stock returned less than a risk-free government T-bill over this "
                "period, so its risk was not rewarded.")
    return (f"Each unit of risk (volatility) earned {s:.2f} units of return above "
            "the risk-free rate.")


def stock_points(sm: Summary, rel_mkt: RelativeMetrics | None, market_name: str,
                 sector_sm: Summary | None = None, sector_name: str | None = None) -> list[str]:
    pts: list[str] = []

    if sm.sharpe >= 2:
        pts.append("A Sharpe above 2 is unusual. It typically comes from a short or "
                   "one-directional stretch and rarely persists for long.")

    if _ok(sm.sortino) and _ok(sm.sharpe) and sm.sharpe > 0:
        if sm.sortino > 1.5 * sm.sharpe:
            pts.append(f"Sortino ({num(sm.sortino)}) is well above Sharpe, so much of the "
                       "volatility came from upward moves rather than falls.")
        elif sm.sortino < 1.2 * sm.sharpe:
            pts.append(f"Sortino ({num(sm.sortino)}) is close to Sharpe, so falls made up a "
                       "large part of the volatility.")

    dd = sm.drawdown
    if dd.max_drawdown < 0:
        line = (f"Its worst fall was {pct(-dd.max_drawdown)}, from the peak on "
                f"{day(dd.peak_date)} to the low on {day(dd.trough_date)}.")
        if dd.recovery_date is not None:
            line += f" It regained that peak {dd.days_to_recover} days after the low."
        else:
            line += (f" It has not yet regained that peak and is "
                     f"{pct(-dd.current_drawdown)} below it today.")
        pts.append(line)

    pts.append(f"On 1 trading day in 20 it lost more than {pct(sm.var95)}; on those "
               f"days the average loss was {pct(sm.cvar95)}.")

    if rel_mkt is not None:
        b = rel_mkt.beta
        if _ok(b):
            if b < 0.8:
                kind = "moved less than the market (defensive)"
            elif b <= 1.2:
                kind = "moved roughly in line with the market"
            else:
                kind = "moved more than the market (aggressive)"
            pts.append(f"Beta of {num(b)} against the {market_name}: the stock {kind}.")
        if _ok(rel_mkt.alpha):
            word = "more" if rel_mkt.alpha >= 0 else "less"
            pts.append(f"Alpha of {pct(rel_mkt.alpha)} a year: it returned {word} than its "
                       "market exposure alone would explain.")
        up, down = rel_mkt.upside_capture, rel_mkt.downside_capture
        if _ok(up) and _ok(down):
            if abs(up - down) < 0.05:
                shape = "a roughly symmetric pattern"
            elif up > down:
                shape = "a favourable pattern: more of the gains than the losses"
            else:
                shape = "an unfavourable pattern: more of the losses than the gains"
            pts.append(f"On the {market_name}'s up days it moved {pct(up, 0)} as much; on its "
                       f"down days {pct(down, 0)} as much, {shape}.")

    if sector_sm is not None and sector_name and _ok(sector_sm.sharpe) and _ok(sm.sharpe):
        diff = sm.sharpe - sector_sm.sharpe
        if abs(diff) < 0.1:
            pts.append(f"Its Sharpe is about the same as the {sector_name} "
                       f"({num(sector_sm.sharpe)}) over the same dates.")
        else:
            word = "higher" if diff > 0 else "lower"
            pts.append(f"Its Sharpe is {word} than the {sector_name} "
                       f"({num(sector_sm.sharpe)}) over the same dates.")

    return pts


def warnings_for(sm: Summary, label: str = "") -> list[str]:
    who = f"{label}: " if label else ""
    out = []
    if sm.low_confidence:
        out.append(f"{who}only {sm.n_returns} trading days of data. Ratios over periods "
                   "this short are noisy, so treat them as indicative.")
    if _ok(sm.stale_share) and sm.stale_share > STALE_DAY_THRESHOLD:
        out.append(f"{who}the price did not change, or nothing traded, on "
                   f"{pct(sm.stale_share, 0)} of days. Thin trading makes the stock look "
                   "less volatile than it is, which inflates Sharpe and Sortino.")
    return out


def compare_points(name_a: str, sa: Summary, name_b: str, sb: Summary,
                   corr: float) -> list[str]:
    pts = []
    if _ok(sa.sharpe) and _ok(sb.sharpe):
        if abs(sa.sharpe - sb.sharpe) < 0.1:
            pts.append(f"{name_a} and {name_b} delivered about the same return per unit of risk.")
        else:
            hi, lo = (name_a, name_b) if sa.sharpe > sb.sharpe else (name_b, name_a)
            pts.append(f"{hi} delivered more return per unit of risk than {lo} over the same dates.")
    if _ok(sa.volatility) and _ok(sb.volatility):
        calmer = name_a if sa.volatility < sb.volatility else name_b
        pts.append(f"{calmer} was the less volatile of the two.")
    da, db = sa.drawdown.max_drawdown, sb.drawdown.max_drawdown
    shallower = name_a if da > db else name_b
    pts.append(f"{shallower} had the shallower worst fall "
               f"({pct(-max(da, db))} against {pct(-min(da, db))}).")
    if _ok(corr):
        if corr >= 0.7:
            c = "move closely together, so holding both adds little diversification"
        elif corr >= 0.4:
            c = "move together to a moderate degree, giving some diversification"
        else:
            c = "move fairly independently, so holding both diversifies well"
        pts.append(f"Daily returns have a correlation of {num(corr)}: the two {c}.")
    return pts
