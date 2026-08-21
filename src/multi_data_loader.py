"""Update and validate the five market datasets used by V28 live features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "DIA": "DIA",
    "VIX": "^VIX",
}

START_DATE = "2015-01-01"
DATA_DIR = Path("data")
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def clean_market_data(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if data.empty:
        raise RuntimeError(f"No data downloaded for {name}.")

    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = [column for column in REQUIRED_COLUMNS[1:] if column not in data.columns]
    if missing:
        raise RuntimeError(f"{name} download is missing columns: {missing}")

    data = data[REQUIRED_COLUMNS[1:]]
    data.index.name = "Date"
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()

    for column in REQUIRED_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    bad_rows = data[data[REQUIRED_COLUMNS].isna().any(axis=1)]
    if len(bad_rows):
        examples = bad_rows["Date"].dt.strftime("%Y-%m-%d").head(5).tolist()
        raise RuntimeError(f"{name} contains malformed rows after download: {examples}")

    data = (
        data
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if len(data) == 0:
        raise RuntimeError(f"{name} has no valid market rows after cleaning.")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise RuntimeError(f"{name} contains non-positive OHLC prices.")
    if (data["Volume"] < 0).any():
        raise RuntimeError(f"{name} contains negative volume.")
    if not data["Date"].is_monotonic_increasing:
        raise RuntimeError(f"{name} dates are not sorted after cleaning.")
    return data


def download_ticker(name: str, ticker: str) -> pd.DataFrame:
    print(f"Downloading {name} ({ticker})...")
    raw = yf.download(
        ticker,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    data = clean_market_data(raw, name)
    output_file = DATA_DIR / f"{name}.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)
    print(f"Saved {len(data)} rows to {output_file}")
    print(f"{name} latest: {data['Date'].iloc[-1].date()}")
    return data


def validate_synchronization(results: dict[str, pd.DataFrame]) -> None:
    latest = {name: data["Date"].iloc[-1].date().isoformat() for name, data in results.items()}
    dates = set(latest.values())
    print()
    for name in TICKERS:
        print(f"{name} latest: {latest[name]}")
    print()
    if len(dates) == 1:
        print("All datasets synchronized.")
        return
    newest = max(dates)
    stale = {name: date for name, date in latest.items() if date != newest}
    raise RuntimeError(f"Market datasets are not synchronized. Newest: {newest}; stale: {stale}")


def main() -> None:
    print()
    print("=" * 60)
    print("MARKET DATA UPDATE")
    print("=" * 60)
    print("PAPER / RESEARCH PIPELINE ONLY")
    print()

    results = {}
    for name, ticker in TICKERS.items():
        results[name] = download_ticker(name, ticker)
        print()

    validate_synchronization(results)


if __name__ == "__main__":
    main()
