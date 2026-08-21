import pandas as pd
import numpy as np
from pathlib import Path

V9_FILE = Path("data/market_features_v9.csv")
SPY_FILE = Path("data/SPY.csv")
QQQ_FILE = Path("data/QQQ.csv")
IWM_FILE = Path("data/IWM.csv")
DIA_FILE = Path("data/DIA.csv")
VIX_FILE = Path("data/VIX.csv")

OUTPUT_FILE = Path("data/market_features_v19.csv")


def load_price(path):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    return (
        df
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


def add_market_features(df, prefix):

    close = df["Close"]

    df[f"{prefix}_return_5d"] = (
        close / close.shift(5) - 1
    )

    df[f"{prefix}_return_20d"] = (
        close / close.shift(20) - 1
    )

    df[f"{prefix}_return_60d"] = (
        close / close.shift(60) - 1
    )

    df[f"{prefix}_sma20"] = close.rolling(20).mean()
    df[f"{prefix}_sma50"] = close.rolling(50).mean()
    df[f"{prefix}_sma200"] = close.rolling(200).mean()

    df[f"{prefix}_distance_sma20"] = (
        close / df[f"{prefix}_sma20"] - 1
    )

    df[f"{prefix}_distance_sma50"] = (
        close / df[f"{prefix}_sma50"] - 1
    )

    df[f"{prefix}_distance_sma200"] = (
        close / df[f"{prefix}_sma200"] - 1
    )

    return df


def main():

    print("Loading V9 features...")

    data = pd.read_csv(V9_FILE)
    data["Date"] = pd.to_datetime(data["Date"])

    spy = load_price(SPY_FILE)
    qqq = load_price(QQQ_FILE)
    iwm = load_price(IWM_FILE)
    dia = load_price(DIA_FILE)
    vix = load_price(VIX_FILE)

    spy = add_market_features(spy, "SPY")
    qqq = add_market_features(qqq, "QQQ")
    iwm = add_market_features(iwm, "IWM")
    dia = add_market_features(dia, "DIA")
    vix = add_market_features(vix, "VIX")

    # --------------------------------------------------
    # REGIME FEATURES
    # --------------------------------------------------

    regime = spy[[
        "Date",
        "SPY_return_5d",
        "SPY_return_20d",
        "SPY_return_60d",
        "SPY_distance_sma20",
        "SPY_distance_sma50",
        "SPY_distance_sma200"
    ]].copy()

    qqq_map = qqq.set_index("Date")
    iwm_map = iwm.set_index("Date")
    dia_map = dia.set_index("Date")
    vix_map = vix.set_index("Date")

    regime = regime.merge(
        qqq_map[["Close", "QQQ_return_20d"]],
        left_on="Date",
        right_index=True,
        how="left"
    )
    regime.rename(columns={"Close": "QQQ_close"}, inplace=True)

    regime = regime.merge(
        iwm_map[["Close", "IWM_return_20d"]],
        left_on="Date",
        right_index=True,
        how="left"
    )
    regime.rename(columns={"Close": "IWM_close"}, inplace=True)

    regime = regime.merge(
        dia_map[["Close", "DIA_return_20d"]],
        left_on="Date",
        right_index=True,
        how="left"
    )
    regime.rename(columns={"Close": "DIA_close"}, inplace=True)

    regime = regime.merge(
        vix_map[[
            "Close",
            "VIX_return_5d",
            "VIX_return_20d",
            "VIX_distance_sma20",
            "VIX_distance_sma50"
        ]],
        left_on="Date",
        right_index=True,
        how="left"
    )
    regime.rename(columns={"Close": "VIX_level"}, inplace=True)

    # Relative strength
    regime["SPY_vs_QQQ_20d"] = (
        regime["SPY_return_20d"]
        - regime["QQQ_return_20d"]
    )

    regime["SPY_vs_IWM_20d"] = (
        regime["SPY_return_20d"]
        - regime["IWM_return_20d"]
    )

    regime["SPY_vs_DIA_20d"] = (
        regime["SPY_return_20d"]
        - regime["DIA_return_20d"]
    )

    # Trend regime
    regime["SPY_above_sma200"] = (
        regime["SPY_distance_sma200"] > 0
    ).astype(int)

    regime["SPY_above_sma50"] = (
        regime["SPY_distance_sma50"] > 0
    ).astype(int)

    regime["strong_bull_regime"] = (
        (regime["SPY_distance_sma200"] > 0.05)
        &
        (regime["SPY_distance_sma50"] > 0.02)
    ).astype(int)

    regime["strong_bear_regime"] = (
        (regime["SPY_distance_sma200"] < -0.05)
        &
        (regime["SPY_distance_sma50"] < -0.02)
    ).astype(int)

    # VIX regime
    regime["high_vix"] = (
        regime["VIX_level"]
        > regime["VIX_level"].rolling(60).median()
    ).astype(int)

    regime["vix_rising"] = (
        regime["VIX_return_5d"] > 0
    ).astype(int)

    # Market agreement
    regime["market_agreement"] = (
        (regime["SPY_return_20d"] > 0).astype(int)
        +
        (regime["QQQ_return_20d"] > 0).astype(int)
        +
        (regime["IWM_return_20d"] > 0).astype(int)
        +
        (regime["DIA_return_20d"] > 0).astype(int)
    )

    # --------------------------------------------------
    # MERGE FEATURES
    # --------------------------------------------------

    data = data.merge(
        regime,
        on="Date",
        how="inner",
        suffixes=("", "_new")
    )

    # --------------------------------------------------
    # V19 TARGETS
    #
    # Every horizon gets its own future return.
    # These are calculated only from future SPY prices.
    # --------------------------------------------------

    spy_prices = spy[["Date", "Close"]].copy()

    spy_prices["future_1d_return"] = (
        spy_prices["Close"].shift(-1)
        / spy_prices["Close"]
        - 1
    )

    spy_prices["future_3d_return"] = (
        spy_prices["Close"].shift(-3)
        / spy_prices["Close"]
        - 1
    )

    spy_prices["future_5d_return"] = (
        spy_prices["Close"].shift(-5)
        / spy_prices["Close"]
        - 1
    )

    spy_prices["future_10d_return"] = (
        spy_prices["Close"].shift(-10)
        / spy_prices["Close"]
        - 1
    )

    data = data.merge(
        spy_prices[[
            "Date",
            "future_1d_return",
            "future_3d_return",
            "future_5d_return",
            "future_10d_return"
        ]],
        on="Date",
        how="left"
    )

    # Remove rows with unavailable feature/target values.
    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    feature_count = len([
        c for c in data.columns
        if c not in [
            "Date",
            "target",
            "future_1d_return",
            "future_3d_return",
            "future_5d_return",
            "future_10d_return"
        ]
    ])

    print()
    print(f"Saved {len(data)} rows")
    print(f"Feature count: {feature_count}")

    print(
        f"Date range: "
        f"{data['Date'].min().date()} → "
        f"{data['Date'].max().date()}"
    )

    print()
    print("Targets:")

    for target in [
        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return"
    ]:
        print()
        print(target)
        print(
            data[target].describe()
        )


if __name__ == "__main__":
    main()