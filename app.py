"""
Sharpe ratio and risk dashboard for Indian equities (NSE and BSE).

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import io
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import benchmarks as bm
import data
import interpret as tx
import metrics as m

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

DEFAULT_RF = 5.5   # % a year. 91-day T-bill cut-off yield was about 5.5-5.6% in Oct 2026.
IST = ZoneInfo("Asia/Kolkata")

INK = "#1C2B39"
STOCK = "#2F4C9A"     # indigo: the stock (or stock A)
STOCK_B = "#12807A"   # teal: stock B in comparisons
MARKET = "#8C97A3"    # grey: Nifty 50
SECTOR = "#B98A2B"    # ochre: sector index
LOSS = "#B23A2E"
GRID = "#DCE2E6"

st.set_page_config(
    page_title="Sharpe ratio for Indian stocks",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
      .hero {{ display: flex; gap: 2.5rem; align-items: flex-end; flex-wrap: wrap;
               padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid {GRID};
               margin-bottom: 1.25rem; }}
      .hero-figure {{ font-family: 'Newsreader', Georgia, serif; font-size: 5.2rem;
                      line-height: 0.9; font-weight: 500; color: {INK};
                      font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
      .hero-figure.neg {{ color: {LOSS}; }}
      .hero-text {{ max-width: 34rem; padding-bottom: 0.35rem; }}
      .hero-band {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.7rem; line-height: 1.2;
                    color: {INK}; margin: 0 0 0.25rem 0; }}
      .hero-sub {{ color: #4A5866; font-size: 0.98rem; margin: 0; line-height: 1.5; }}
      .who {{ color: #4A5866; font-size: 0.95rem; margin-bottom: 0.2rem; }}
      .pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
               gap: 1.5rem; padding-bottom: 1.25rem; border-bottom: 1px solid {GRID};
               margin-bottom: 1.25rem; }}
      .pair .hero-figure {{ font-size: 4rem; }}
      .swatch {{ display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 2px;
                 margin-right: 0.4rem; vertical-align: baseline; }}
      .footnote {{ color: #5E6B78; font-size: 0.85rem; line-height: 1.5;
                   border-top: 1px solid {GRID}; margin-top: 2.5rem; padding-top: 1rem; }}
      [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached data access (shared across all users of the app)
# --------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_prices(raw: str, exchange: str, start: dt.date, end: dt.date) -> data.PriceData:
    return data.load_prices(raw, exchange, start, end)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_profile(symbol: str) -> data.Profile:
    return data.get_source().profile(symbol)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_index(symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    return data.get_source().history(symbol, start, end).close


def benchmark_prices(name: str, start: dt.date, end: dt.date) -> tuple[str, pd.Series, str | None]:
    """Fetch a benchmark, falling back to Nifty 50 if the provider has no
    history for it. Returns (name actually used, prices, note)."""
    series = cached_index(bm.BENCHMARKS[name], start, end)
    if len(series) >= data.USABLE_ROWS or name == bm.MARKET_BENCHMARK:
        return name, series, None
    fallback = bm.MARKET_BENCHMARK
    note = (f"The data provider has no history for the {name} index right now, "
            f"so the {fallback} is used as the sector benchmark instead.")
    return fallback, cached_index(bm.BENCHMARKS[fallback], start, end), note


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def from_start(prices: pd.Series, start: pd.Timestamp) -> pd.Series:
    return prices.loc[prices.index >= start]


def growth_of_100(prices: pd.Series) -> pd.Series:
    return prices / prices.iloc[0] * 100


def chart_layout(fig: go.Figure, y_title: str, y_format: str | None = None) -> go.Figure:
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False, tickformat=y_format),
        xaxis=dict(gridcolor=GRID, showgrid=False),
    )
    return fig


def line(fig, s: pd.Series, name: str, color: str, width=2.2, dash=None, fmt=".1f"):
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        hovertemplate=f"%{{y:{fmt}}}",
    ))


def growth_chart(series: list[tuple[str, pd.Series, str]]) -> go.Figure:
    fig = go.Figure()
    for i, (name, p, color) in enumerate(series):
        line(fig, growth_of_100(p), name, color, width=2.4 if i == 0 else 1.6)
    fig.add_hline(y=100, line=dict(color=GRID, width=1))
    return chart_layout(fig, "Value of ₹100 invested")


def drawdown_chart(series: list[tuple[str, pd.Series, str]]) -> go.Figure:
    fig = go.Figure()
    for i, (name, p, color) in enumerate(series):
        dd = m.drawdown_series(p)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name, mode="lines",
            line=dict(color=color, width=1.8),
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(178,58,46,0.12)" if i == 0 else None,
            hovertemplate="%{y:.1%}",
        ))
    return chart_layout(fig, "Below previous peak", ".0%")


def rolling_chart(series: list[tuple[str, pd.Series, str]], rf: float) -> go.Figure | None:
    window = m.rolling_window_for(min(len(p) - 1 for _, p, _ in series))
    if window is None:
        return None
    fig = go.Figure()
    for name, p, color in series:
        rs = m.rolling_sharpe(m.daily_returns(p), rf, window).dropna()
        line(fig, rs, name, color, fmt=".2f")
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig.add_hline(y=1, line=dict(color=GRID, width=1, dash="dash"))
    fig = chart_layout(fig, f"Sharpe over trailing {window} trading days")
    fig.update_layout(meta=window)
    return fig


def metrics_table(cols: dict[str, m.Summary]) -> pd.DataFrame:
    rows = {
        "Sharpe ratio": lambda s: tx.num(s.sharpe),
        "Sortino ratio": lambda s: tx.num(s.sortino),
        "Calmar ratio": lambda s: tx.num(s.calmar),
        "Total return": lambda s: tx.pct(s.total_return),
        "Annualised return (CAGR)": lambda s: tx.pct(s.cagr),
        "Annualised volatility": lambda s: tx.pct(s.volatility),
        "Maximum drawdown": lambda s: tx.pct(s.drawdown.max_drawdown),
        "Current distance from peak": lambda s: tx.pct(s.drawdown.current_drawdown),
        "1-day VaR (95%)": lambda s: tx.pct(s.var95, 2),
        "Expected shortfall (95%)": lambda s: tx.pct(s.cvar95, 2),
        "Trading days": lambda s: str(s.n_returns),
    }
    return pd.DataFrame({c: [f(s) for f in rows.values()] for c, s in cols.items()},
                        index=list(rows.keys()))


def relative_table(cols: dict[str, m.RelativeMetrics]) -> pd.DataFrame:
    rows = {
        "Beta": lambda r: tx.num(r.beta),
        "Alpha (annualised)": lambda r: tx.pct(r.alpha),
        "Correlation": lambda r: tx.num(r.correlation),
        "Upside capture": lambda r: tx.pct(r.upside_capture, 0),
        "Downside capture": lambda r: tx.pct(r.downside_capture, 0),
        "Tracking error": lambda r: tx.pct(r.tracking_error),
        "Information ratio": lambda r: tx.num(r.information_ratio),
    }
    return pd.DataFrame({c: [f(r) for f in rows.values()] for c, r in cols.items()},
                        index=list(rows.keys()))


def csv_bytes(tables: dict[str, pd.DataFrame], prices: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    for title, df in tables.items():
        buf.write(f"{title}\n")
        df.to_csv(buf)
        buf.write("\n")
    buf.write("Daily adjusted closing prices\n")
    prices.to_csv(buf, date_format="%Y-%m-%d")
    return buf.getvalue().encode("utf-8")


def hero(value: float, title_html: str, sub: str) -> None:
    neg = " neg" if tx._ok(value) and value < 0 else ""
    st.markdown(
        f"""<div class="hero">
              <div>
                <div class="who">Sharpe ratio</div>
                <div class="hero-figure{neg}">{tx.num(value)}</div>
              </div>
              <div class="hero-text">
                <div class="hero-band">{title_html}</div>
                <div class="hero-sub">{sub}</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


METHOD = f"""
**Data.** Daily closing prices from Yahoo Finance, adjusted for splits, bonuses and
dividends, so returns include dividends. Benchmark indices (Nifty 50 and sector
indices) are price indices without dividends, which flatters the stock slightly in
any comparison, by roughly the index's dividend yield (about 1-1.5% a year for the Nifty 50).

**Period.** From the first trading day on or after the buy date to the latest close.
Gains or losses on the buy day itself, between the purchase price and that day's
close, are not included.

**Daily returns.** r = today's close / yesterday's close - 1.

**Risk-free rate.** The annual rate you enter, converted to a daily rate by
compounding: (1 + rate)^(1/252) - 1. The default is the 91-day Treasury bill yield.

**Sharpe ratio.** Average daily return above the risk-free rate, divided by the
standard deviation of daily returns, multiplied by √252 to annualise.

**Sortino ratio.** As Sharpe, but the denominator counts only days that returned
less than the risk-free rate (downside deviation).

**Calmar ratio.** Annualised return divided by the maximum drawdown.

**Annualised return.** Compound annual growth over calendar days. Over periods
shorter than a year this extrapolates, so read it alongside the total return.

**Volatility.** Standard deviation of daily returns × √252.

**Maximum drawdown.** The largest fall from a previous peak to a later low.

**VaR and expected shortfall.** Historical: the loss exceeded on the worst 5% of
days, and the average loss on those days.

**Beta, alpha, capture, correlation.** Calculated only on days both the stock and
the index traded. Alpha is Jensen's alpha, annualised. Upside capture is the stock's
average return on the index's up days divided by the index's average return on those
days; downside capture is the same for down days.

**Rolling Sharpe.** Sharpe over a trailing 90-day window (63 days for periods under
about nine months; not shown for periods under about six months).

**Sector benchmark.** Picked from Yahoo's industry and sector tags, mapped to the
closest Nifty index. The match is approximate; choose another index from the list
if it does not fit.

**Thresholds.** Below {m.MINIMUM_DAYS} trading days nothing is calculated. Below
{m.LOW_CONFIDENCE_DAYS} the figures are shown with a warning. A liquidity warning
appears when the price is unchanged or nothing trades on more than
{int(m.STALE_DAY_THRESHOLD * 100)}% of days.

**Comparisons.** Both stocks are measured from the same start date. If one listed
later, both start from its first trading day. Comparisons use the Nifty 50 as the benchmark.

All ratios use 252 trading days a year, the global convention, so results are
comparable with other platforms.
"""

FOOTNOTE = """
<div class="footnote">
For information and education only. This tool reports historical risk and return
figures; it does not give investment advice or a recommendation to buy, sell or hold any
security. Past performance does not indicate future results. Prices come from Yahoo Finance
and may be delayed, incomplete or wrong; check important figures against your broker or the
exchange.
</div>
"""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

today = dt.datetime.now(IST).date()
default_start = today - dt.timedelta(days=365)

st.title("Risk-adjusted return, from your buy date to today")
st.write("Enter an NSE or BSE stock and the date you bought it. The tool measures how "
         "well the return has paid for the risk taken, and how that compares with the market.")

mode = st.radio("What do you want to check?", ["One stock", "Compare two stocks"],
                horizontal=True, label_visibility="collapsed")

sector_options = ["Pick automatically"] + [n for n in bm.BENCHMARKS if n != bm.MARKET_BENCHMARK]

with st.form("inputs"):
    if mode == "One stock":
        c1, c2, c3 = st.columns([1.6, 0.8, 1.1])
        t1 = c1.text_input("Ticker", placeholder="e.g. RELIANCE, TCS, HDFCBANK")
        ex1 = c2.selectbox("Exchange", ["NSE", "BSE"])
        start = c3.date_input("Buy date", value=default_start,
                              min_value=dt.date(2000, 1, 1), max_value=today - dt.timedelta(days=1),
                              format="DD/MM/YYYY")
        c4, c5 = st.columns([1, 1.5])
        rf_pct = c4.number_input("Risk-free rate (% a year)", 0.0, 20.0, DEFAULT_RF, 0.05,
                                 help="Defaults to the 91-day Treasury bill yield.")
        sector_choice = c5.selectbox("Sector benchmark", sector_options,
                                     help="Picked from the company's industry. Change it if the match looks wrong.")
        t2 = ex2 = None
    else:
        c1, c2, c3, c4 = st.columns([1.4, 0.7, 1.4, 0.7])
        t1 = c1.text_input("First ticker", placeholder="e.g. TCS")
        ex1 = c2.selectbox("Exchange ", ["NSE", "BSE"])
        t2 = c3.text_input("Second ticker", placeholder="e.g. INFY")
        ex2 = c4.selectbox("Exchange  ", ["NSE", "BSE"])
        c5, c6 = st.columns(2)
        start = c5.date_input("Start date (both stocks)", value=default_start,
                              min_value=dt.date(2000, 1, 1), max_value=today - dt.timedelta(days=1),
                              format="DD/MM/YYYY")
        rf_pct = c6.number_input("Risk-free rate (% a year)", 0.0, 20.0, DEFAULT_RF, 0.05,
                                 help="Defaults to the 91-day Treasury bill yield.")
        sector_choice = None
    submitted = st.form_submit_button("Calculate", type="primary")

if submitted:
    st.session_state["run"] = dict(mode=mode, t1=t1, ex1=ex1, t2=t2, ex2=ex2,
                                   start=start, rf=rf_pct / 100, sector=sector_choice)

run = st.session_state.get("run")
if not run or run["mode"] != mode:
    st.info("Enter a ticker and a date, then select Calculate.")
    st.markdown(FOOTNOTE, unsafe_allow_html=True)
    st.stop()

rf = run["rf"]
start = run["start"]


# --------------------------------------------------------------------------
# One stock
# --------------------------------------------------------------------------

def show_one_stock() -> None:
    with st.spinner("Fetching prices and calculating"):
        try:
            pdata = cached_prices(run["t1"], run["ex1"], start, today)
        except data.DataError as e:
            st.error(str(e))
            return
        prices = pdata.close
        try:
            sm = m.summarise(prices, rf, pdata.volume)
        except ValueError as e:
            st.error(str(e))
            return

        profile = cached_profile(pdata.symbol)
        first_day = prices.index[0]

        mkt = from_start(cached_index(bm.BENCHMARKS[bm.MARKET_BENCHMARK], start, today), first_day)
        mkt_sm = m.summarise(mkt, rf) if len(mkt) > m.MINIMUM_DAYS else None
        rel_mkt = (m.relative_metrics(m.daily_returns(prices), m.daily_returns(mkt), rf)
                   if mkt_sm else None)

        if run["sector"] == "Pick automatically":
            wanted, reason = bm.suggest_sector_benchmark(profile.sector, profile.industry)
        else:
            wanted, reason = run["sector"], "chosen by you"
        sec_name, sec_raw, sec_note = benchmark_prices(wanted, start, today)
        sec = from_start(sec_raw, first_day)
        show_sector = sec_name != bm.MARKET_BENCHMARK and len(sec) > m.MINIMUM_DAYS
        sec_sm = m.summarise(sec, rf) if show_sector else None
        rel_sec = (m.relative_metrics(m.daily_returns(prices), m.daily_returns(sec), rf)
                   if show_sector else None)

    # Heading
    st.subheader(f"{profile.name}")
    st.caption(f"{pdata.symbol}   |   {tx.day(sm.start)} to {tx.day(sm.end)}   |   "
               f"{sm.n_returns} trading days   |   risk-free rate {rf * 100:.2f}%")

    hero(sm.sharpe, tx.sharpe_band(sm.sharpe), tx.sharpe_sentence(sm.sharpe))

    # Notes the user should see before reading numbers
    notes = []
    if pdata.note:
        notes.append(pdata.note)
    if first_day.date() > start:
        if (first_day.date() - start).days > 7:
            notes.append(f"Price history starts on {tx.day(first_day)}, later than the buy date "
                         "(listing date or the data provider's history limit), so figures start there.")
        else:
            notes.append(f"{start.strftime('%d %b %Y')} was not a trading day, so the "
                         f"period starts on the next one, {tx.day(first_day)}.")
    if sec_note:
        notes.append(sec_note)
    for n in notes:
        st.info(n)
    for w in tx.warnings_for(sm):
        st.warning(w[0].upper() + w[1:])

    a, b, c, d = st.columns(4)
    a.metric("Sortino ratio", tx.num(sm.sortino))
    b.metric("Annualised return", tx.pct(sm.cagr), help=f"Total return {tx.pct(sm.total_return)}")
    c.metric("Annualised volatility", tx.pct(sm.volatility))
    d.metric("Maximum drawdown", tx.pct(sm.drawdown.max_drawdown))

    series = [(pdata.symbol, prices, STOCK)]
    if mkt_sm:
        series.append((bm.MARKET_BENCHMARK, mkt, MARKET))
    if show_sector:
        series.append((sec_name, sec, SECTOR))

    tabs = st.tabs(["Growth", "Drawdown", "Rolling Sharpe", "All figures", "How it's calculated"])
    with tabs[0]:
        st.plotly_chart(growth_chart(series), width="stretch")
    with tabs[1]:
        st.plotly_chart(drawdown_chart(series), width="stretch")
    with tabs[2]:
        fig = rolling_chart(series, rf)
        if fig is None:
            st.write("The period is too short for a rolling view. It needs about six months of data.")
        else:
            st.plotly_chart(fig, width="stretch")
            st.caption("Shows whether risk-adjusted performance has been improving or "
                       "fading. The dashed line marks a Sharpe of 1.")
    with tabs[3]:
        cols = {pdata.symbol: sm}
        if mkt_sm:
            cols[bm.MARKET_BENCHMARK] = mkt_sm
        if sec_sm:
            cols[sec_name] = sec_sm
        perf = metrics_table(cols)
        st.dataframe(perf, width="stretch")
        rels = {}
        if rel_mkt:
            rels[f"vs {bm.MARKET_BENCHMARK}"] = rel_mkt
        if rel_sec:
            rels[f"vs {sec_name}"] = rel_sec
        rel_df = relative_table(rels) if rels else pd.DataFrame()
        if rels:
            st.markdown(f"**{pdata.symbol} against the benchmarks**")
            st.dataframe(rel_df, width="stretch")
        if show_sector:
            st.caption(f"Sector benchmark: {sec_name} ({reason}).")
        price_df = pd.DataFrame({name: p for name, p, _ in series})
        st.download_button(
            "Download figures and prices (CSV)",
            csv_bytes({"Performance": perf, "Against benchmarks": rel_df}, price_df),
            file_name=f"{pdata.symbol}_{sm.start:%Y%m%d}_{sm.end:%Y%m%d}.csv",
            mime="text/csv",
        )
    with tabs[4]:
        st.markdown(METHOD)

    st.markdown("#### What the figures show")
    for p in tx.stock_points(sm, rel_mkt, bm.MARKET_BENCHMARK, sec_sm,
                             sec_name if show_sector else None):
        st.markdown(f"- {p}")


# --------------------------------------------------------------------------
# Two stocks
# --------------------------------------------------------------------------

def show_comparison() -> None:
    with st.spinner("Fetching prices and calculating"):
        try:
            da = cached_prices(run["t1"], run["ex1"], start, today)
            db = cached_prices(run["t2"], run["ex2"], start, today)
        except data.DataError as e:
            st.error(str(e))
            return
        if da.symbol == db.symbol:
            st.error("Both tickers point to the same stock. Enter two different companies.")
            return

        common_start = max(da.close.index[0], db.close.index[0])
        pa, pb = from_start(da.close, common_start), from_start(db.close, common_start)
        va = da.volume.reindex(pa.index) if da.volume is not None else None
        vb = db.volume.reindex(pb.index) if db.volume is not None else None
        try:
            sa, sb = m.summarise(pa, rf, va), m.summarise(pb, rf, vb)
        except ValueError as e:
            st.error(str(e))
            return

        na, nb = da.symbol, db.symbol
        mkt = from_start(cached_index(bm.BENCHMARKS[bm.MARKET_BENCHMARK], start, today), common_start)
        mkt_sm = m.summarise(mkt, rf) if len(mkt) > m.MINIMUM_DAYS else None
        ra, rb = m.daily_returns(pa), m.daily_returns(pb)
        rel_a = m.relative_metrics(ra, m.daily_returns(mkt), rf) if mkt_sm else None
        rel_b = m.relative_metrics(rb, m.daily_returns(mkt), rf) if mkt_sm else None
        corr = m.relative_metrics(ra, rb, rf).correlation

    st.subheader(f"{na} and {nb}")
    st.caption(f"{tx.day(common_start)} to {tx.day(min(sa.end, sb.end))}   |   "
               f"risk-free rate {rf * 100:.2f}%")

    def block(name, s, color):
        neg = " neg" if tx._ok(s.sharpe) and s.sharpe < 0 else ""
        return (f'<div><div class="who"><span class="swatch" style="background:{color}"></span>{name}</div>'
                f'<div class="hero-figure{neg}">{tx.num(s.sharpe)}</div>'
                f'<div class="hero-sub">{tx.sharpe_band(s.sharpe)}</div></div>')

    st.markdown(f'<div class="pair">{block(na, sa, STOCK)}{block(nb, sb, STOCK_B)}</div>',
                unsafe_allow_html=True)

    for d_ in (da, db):
        if d_.note:
            st.info(d_.note)
    if common_start.date() > start + dt.timedelta(days=7):
        st.info(f"One of the stocks has no prices before {tx.day(common_start)}, so both are "
                "measured from that date to keep the comparison fair.")
    for w in tx.warnings_for(sa, na) + tx.warnings_for(sb, nb):
        st.warning(w)

    series = [(na, pa, STOCK), (nb, pb, STOCK_B)]
    with_mkt = series + ([(bm.MARKET_BENCHMARK, mkt, MARKET)] if mkt_sm else [])

    tabs = st.tabs(["Growth", "Drawdown", "Rolling Sharpe", "All figures", "How it's calculated"])
    with tabs[0]:
        st.plotly_chart(growth_chart(with_mkt), width="stretch")
    with tabs[1]:
        st.plotly_chart(drawdown_chart(series), width="stretch")
    with tabs[2]:
        fig = rolling_chart(series, rf)
        if fig is None:
            st.write("The period is too short for a rolling view. It needs about six months of data.")
        else:
            st.plotly_chart(fig, width="stretch")
    with tabs[3]:
        cols = {na: sa, nb: sb}
        if mkt_sm:
            cols[bm.MARKET_BENCHMARK] = mkt_sm
        perf = metrics_table(cols)
        st.dataframe(perf, width="stretch")
        rel_df = pd.DataFrame()
        if rel_a and rel_b:
            st.markdown(f"**Each stock against the {bm.MARKET_BENCHMARK}**")
            rel_df = relative_table({na: rel_a, nb: rel_b})
            st.dataframe(rel_df, width="stretch")
        st.markdown(f"Correlation of daily returns between {na} and {nb}: **{tx.num(corr)}**")
        price_df = pd.DataFrame({name: p for name, p, _ in with_mkt})
        st.download_button(
            "Download figures and prices (CSV)",
            csv_bytes({"Performance": perf, f"Against {bm.MARKET_BENCHMARK}": rel_df}, price_df),
            file_name=f"{na}_vs_{nb}_{common_start:%Y%m%d}.csv",
            mime="text/csv",
        )
    with tabs[4]:
        st.markdown(METHOD)

    st.markdown("#### What the figures show")
    for p in tx.compare_points(na, sa, nb, sb, corr):
        st.markdown(f"- {p}")


if run["mode"] == "One stock":
    show_one_stock()
else:
    show_comparison()

st.markdown(FOOTNOTE, unsafe_allow_html=True)
