"""Checks the maths against hand-computed answers. Run: python -m pytest tests"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as m  # noqa: E402


def _prices(values, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_daily_rf_compounds_back_to_annual():
    assert (1 + m.daily_rf(0.055)) ** 252 - 1 == pytest.approx(0.055)


def test_sharpe_matches_textbook_formula():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0008, 0.015, 500))
    rf = 0.055
    expected = (r - (1.055 ** (1 / 252) - 1)).mean() / r.std(ddof=1) * math.sqrt(252)
    assert m.sharpe_ratio(r, rf) == pytest.approx(expected)


def test_sortino_ignores_upside_days():
    r = pd.Series([0.02, 0.03, -0.01, 0.04, -0.02])
    ex = r - m.daily_rf(0.0)
    dd = math.sqrt((np.minimum(ex, 0) ** 2).mean())
    assert m.sortino_ratio(r, 0.0) == pytest.approx(ex.mean() / dd * math.sqrt(252))


def test_zero_volatility_gives_nan_not_crash():
    p = _prices([100] * 30)
    assert math.isnan(m.sharpe_ratio(m.daily_returns(p), 0.05))


def test_drawdown_peak_trough_recovery():
    p = _prices([100, 120, 90, 100, 130])
    info = m.drawdown_info(p)
    assert info.max_drawdown == pytest.approx(-0.25)
    assert info.peak_date == p.index[1]
    assert info.trough_date == p.index[2]
    assert info.recovery_date == p.index[4]
    assert info.current_drawdown == 0


def test_unrecovered_drawdown():
    info = m.drawdown_info(_prices([100, 120, 90, 100]))
    assert info.recovery_date is None
    assert info.current_drawdown == pytest.approx(100 / 120 - 1)


def test_var_and_expected_shortfall():
    r = pd.Series(np.linspace(-0.10, 0.09, 20))  # evenly spread returns
    var, cvar = m.value_at_risk(r, 0.95)
    assert var == pytest.approx(-np.percentile(r, 5))
    assert cvar >= var


def test_benchmark_against_itself():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0.0005, 0.01, 300), index=pd.bdate_range("2025-01-01", periods=300))
    rel = m.relative_metrics(b, b, 0.055)
    assert rel.beta == pytest.approx(1.0)
    assert rel.alpha == pytest.approx(0.0, abs=1e-12)
    assert rel.correlation == pytest.approx(1.0)
    assert rel.upside_capture == pytest.approx(1.0)
    assert rel.downside_capture == pytest.approx(1.0)


def test_leveraged_stock_has_beta_two():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2025-01-01", periods=400)
    b = pd.Series(rng.normal(0.0004, 0.01, 400), index=idx)
    s = 2 * b
    rel = m.relative_metrics(s, b, 0.0)
    assert rel.beta == pytest.approx(2.0)
    assert rel.upside_capture == pytest.approx(2.0)
    assert rel.downside_capture == pytest.approx(2.0)


def test_alignment_drops_dates_missing_from_either_side():
    idx = pd.bdate_range("2025-01-01", periods=10)
    s = pd.Series(range(10), index=idx, dtype=float)
    b = s.drop(idx[3])
    assert len(m.align(s, b)) == 9


def test_stale_share_counts_unchanged_closes():
    p = _prices([100, 100, 101, 101, 102])  # 2 of 4 changes are zero
    assert m.stale_day_share(p) == pytest.approx(0.5)


def test_too_few_days_refused():
    with pytest.raises(ValueError):
        m.summarise(_prices(np.linspace(100, 110, 10)), 0.05)


def test_low_confidence_flag():
    rng = np.random.default_rng(3)
    p = _prices(100 * np.cumprod(1 + rng.normal(0, 0.01, 40)))
    assert m.summarise(p, 0.05).low_confidence is True


def test_cagr_one_year_doubling():
    p = pd.Series([100.0, 200.0], index=pd.to_datetime(["2025-01-01", "2026-01-01"]))
    assert m.cagr(p) == pytest.approx(2 ** (365.25 / 365) - 1)
