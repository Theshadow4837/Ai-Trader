import yfinance as yf
from pathlib import Path


TICKER = "SPY"
START_DATE = "2015-01-01"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_data():
    print(f"Downloading {TICKER} historical data...")

    data = yf.download(
        TICKER,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if data.empty:
        raise RuntimeError("No market data was downloaded.")

    data = data[["Open", "High", "Low", "Close", "Volume"]]

    data.index.name = "Date"
    data.reset_index(inplace=True)

    output_file = DATA_DIR / f"{TICKER}.csv"
    data.to_csv(output_file, index=False)

    print(f"Saved {len(data)} rows to {output_file}")


if __name__ == "__main__":
    download_data()