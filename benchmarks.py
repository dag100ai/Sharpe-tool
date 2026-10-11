"""
Benchmark indices and the rules that pick a sector benchmark for a stock.

The sector match is approximate: Yahoo tags each company with a sector and
an industry, and these tables map those tags to the closest Nifty index.
The user can always override the choice in the app.

Indices are listed whether or not the data provider currently has history
for them; the app checks at run time and falls back to Nifty 50 (with a
visible note) if an index has no usable data.
"""

from __future__ import annotations

MARKET_BENCHMARK = "Nifty 50"

BENCHMARKS: dict[str, str] = {
    # Broad market
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Midcap 150": "NIFTYMIDCAP150.NS",
    "Nifty Midcap 50": "^NSEMDCP50",
    "BSE 500": "BSE-500.BO",
    # Sectors
    "Nifty Bank": "^NSEBANK",
    "Nifty PSU Bank": "^CNXPSUBANK",
    "Nifty Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Nifty IT": "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Healthcare": "NIFTY_HEALTHCARE.NS",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Oil & Gas": "NIFTY_OIL_AND_GAS.NS",
    "Nifty Infrastructure": "^CNXINFRA",
    "Nifty Media": "^CNXMEDIA",
    "Nifty Consumer Durables": "NIFTY_CONSR_DURBL.NS",
    "Nifty Commodities": "^CNXCMDT",
}

# Checked first: Yahoo industry -> index. Matching is case-insensitive and
# on substrings, so "Banks - Regional" matches "banks".
INDUSTRY_RULES: list[tuple[str, str]] = [
    ("banks", "Nifty Bank"),
    ("information technology", "Nifty IT"),
    ("software", "Nifty IT"),
    ("drug manufacturers", "Nifty Pharma"),
    ("biotechnology", "Nifty Pharma"),
    ("medical", "Nifty Healthcare"),
    ("diagnostics", "Nifty Healthcare"),
    ("healthcare plans", "Nifty Healthcare"),
    ("auto", "Nifty Auto"),
    ("steel", "Nifty Metal"),
    ("aluminum", "Nifty Metal"),
    ("copper", "Nifty Metal"),
    ("industrial metals", "Nifty Metal"),
    ("coking coal", "Nifty Metal"),
    ("real estate", "Nifty Realty"),
    ("oil & gas", "Nifty Oil & Gas"),
    ("utilities", "Nifty Energy"),
    ("building materials", "Nifty Infrastructure"),
    ("engineering & construction", "Nifty Infrastructure"),
    ("infrastructure", "Nifty Infrastructure"),
    ("entertainment", "Nifty Media"),
    ("broadcasting", "Nifty Media"),
    ("publishing", "Nifty Media"),
    ("household & personal", "Nifty FMCG"),
    ("packaged foods", "Nifty FMCG"),
    ("tobacco", "Nifty FMCG"),
    ("beverages", "Nifty FMCG"),
    ("confectioners", "Nifty FMCG"),
    ("furnishings", "Nifty Consumer Durables"),
    ("appliances", "Nifty Consumer Durables"),
    ("consumer electronics", "Nifty Consumer Durables"),
    ("luxury goods", "Nifty Consumer Durables"),
    ("chemicals", "Nifty Commodities"),
    ("credit services", "Nifty Financial Services"),
    ("capital markets", "Nifty Financial Services"),
    ("insurance", "Nifty Financial Services"),
    ("asset management", "Nifty Financial Services"),
    ("mortgage finance", "Nifty Financial Services"),
]

# Used when no industry rule matches.
SECTOR_RULES: dict[str, str] = {
    "technology": "Nifty IT",
    "healthcare": "Nifty Healthcare",
    "financial services": "Nifty Financial Services",
    "energy": "Nifty Energy",
    "utilities": "Nifty Energy",
    "basic materials": "Nifty Commodities",
    "real estate": "Nifty Realty",
    "consumer defensive": "Nifty FMCG",
    "communication services": "Nifty Media",
}


def suggest_sector_benchmark(sector: str | None, industry: str | None) -> tuple[str, str]:
    """Return (benchmark name, short reason)."""
    ind = (industry or "").lower()
    for key, bench in INDUSTRY_RULES:
        if key in ind:
            return bench, f"matched on industry: {industry}"
    sec = (sector or "").lower()
    if sec in SECTOR_RULES:
        return SECTOR_RULES[sec], f"matched on sector: {sector}"
    return MARKET_BENCHMARK, "no close sector index, using the broad market"
