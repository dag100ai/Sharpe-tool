# Sharpe ratio for Indian stocks

A web tool that measures a stock's risk-adjusted return from a buy date to
today. Enter an NSE or BSE ticker and a date; it fetches daily prices and
reports Sharpe, Sortino, Calmar, drawdowns, Value at Risk, beta, alpha and
capture ratios, against the Nifty 50 and the closest Nifty sector index.
A second mode compares two stocks over the same dates.

## Files

| File | What it does |
|---|---|
| `app.py` | The dashboard (Streamlit) |
| `metrics.py` | All calculations. Pure functions, no data fetching |
| `data.py` | Fetches prices. The only file that knows about Yahoo Finance |
| `benchmarks.py` | Index list and the stock-to-sector-index matching rules |
| `interpret.py` | Plain-language reading of the figures (fixed rules) |
| `tests/test_metrics.py` | Checks the maths against hand-computed answers |
| `.streamlit/config.toml` | Colours and fonts |
| `requirements.txt` | Python packages, pinned to the tested versions |

## Run it on your own computer

You need Python 3.10 or newer.

```bash
cd sharpe-tool
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens at http://localhost:8501.

To check the maths: `pip install pytest` then `python -m pytest tests`.

## Publish it with a shareable link (free)

1. Create a free account at https://github.com.
2. Create a new public repository (for example `sharpe-tool`) and upload every
   file and folder from this project, keeping the same structure. The
   `.streamlit` folder starts with a dot; make sure it is included.
3. Go to https://share.streamlit.io and sign in with your GitHub account.
4. Select **Create app**, choose your repository, branch `main`, and main file
   `app.py`. Under advanced settings choose Python 3.12. Deploy.
5. After a few minutes you get a link like `https://<name>.streamlit.app`.
   Anyone with the link can use it.

To update the app later, change the files on GitHub; the app redeploys itself.

Free-tier apps go to sleep after a few days without visitors. The first person
to open a sleeping app waits about a minute while it wakes up.

## Things to maintain

* **Risk-free rate default.** `DEFAULT_RF` at the top of `app.py` (5.5%). Update
  it when the 91-day T-bill yield moves meaningfully; RBI publishes auction
  results weekly. Users can also type their own rate.
* **Data provider.** Yahoo Finance is unofficial and can change without notice.
  If it breaks, upgrade the package (`yfinance` in `requirements.txt`). To move
  to another provider, write a new class in `data.py` with the same three
  methods as `YahooSource` and return it from `get_source()`.
* **Sector indices.** At the time of building, Yahoo returned full history for
  the Nifty 50, Sensex, Nifty Bank, Nifty IT, Nifty Pharma, Nifty Midcap 150,
  Nifty Midcap 50 and BSE 500, but only a single latest quote for most other
  sector indices (Auto, FMCG, Metal, Realty and others). The app checks every
  time and falls back to the Nifty 50, with a visible note, when an index has
  no history. If Yahoo fills those gaps, they start working automatically.
* **BSE listings.** Yahoo's BSE history is patchy. When a `.BO` symbol has no
  history the app uses the same company's NSE prices and says so on screen.

## Known limits

* Benchmarks are price indices without dividends; stock returns include
  dividends. This flatters the stock by roughly the index dividend yield.
* Returns run from the buy-day close, not the actual purchase price.
* Shared hosting means every user's requests go through one server. Results
  are cached for six hours, which keeps load on Yahoo low for popular tickers,
  but very heavy use could hit Yahoo's rate limits.

For information and education only; not investment advice.
