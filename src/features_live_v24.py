import pandas as pd
import numpy as np
from pathlib import Path

from v28_validation import compare_feature_schema, validate_feature_schema, validate_feature_values
from features_v9 import TICKERS, add_market_features as add_v9_market_features, load_market

# ============================================================
# V24 LIVE FEATURE GENERATOR
#
# IMPORTANT:
# - Uses the same market inputs as V14
# - Does NOT require future returns
# - Does NOT overwrite market_features_v14.csv
# - Produces the latest feature rows for V24
# ============================================================

SPY_FILE = Path("data/SPY.csv")
QQQ_FILE = Path("data/QQQ.csv")
IWM_FILE = Path("data/IWM.csv")
DIA_FILE = Path("data/DIA.csv")
VIX_FILE = Path("data/VIX.csv")

OUTPUT_FILE = Path("data/live_features_v24.csv")
REFERENCE_FEATURE_FILE = Path("data/market_features_v14.csv")


def load_price(path):

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    return (
        df
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


def build_base_v9_live_features():
    """Rebuild the original V9 feature set from raw market data, without targets."""
    combined = None

    for ticker in TICKERS:
        data = add_v9_market_features(
            load_market(ticker),
            ticker
        )

        keep = ["Date"] + [
            column
            for column in data.columns
            if column.startswith(f"{ticker}_")
        ]

        data = data[keep]

        if combined is None:
            combined = data
        else:
            combined = combined.merge(
                data,
                on="Date",
                how="inner"
            )

    return (
        combined
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def add_market_features(df, prefix):

    df = df.copy()

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

    df[f"{prefix}_sma20"] = (
        close.rolling(20).mean()
    )

    df[f"{prefix}_sma50"] = (
        close.rolling(50).mean()
    )

    df[f"{prefix}_sma200"] = (
        close.rolling(200).mean()
    )

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

    print()
    print("================================")
    print("      V24 LIVE FEATURES")
    print("================================")

    data = build_base_v9_live_features()

    # --------------------------------------------------------
    # Load current market data
    # --------------------------------------------------------

    spy = add_market_features(
        load_price(SPY_FILE),
        "SPY"
    )

    qqq = add_market_features(
        load_price(QQQ_FILE),
        "QQQ"
    )

    iwm = add_market_features(
        load_price(IWM_FILE),
        "IWM"
    )

    dia = add_market_features(
        load_price(DIA_FILE),
        "DIA"
    )

    vix = add_market_features(
        load_price(VIX_FILE),
        "VIX"
    )

    # --------------------------------------------------------
    # Build regime dataframe
    # --------------------------------------------------------

    regime = spy[[
        "Date",
        "SPY_return_5d",
        "SPY_return_20d",
        "SPY_return_60d",
        "SPY_distance_sma20",
        "SPY_distance_sma50",
        "SPY_distance_sma200"
    ]].copy()

    # --------------------------------------------------------
    # QQQ
    # --------------------------------------------------------

    qqq_map = qqq.set_index("Date")

    regime = regime.merge(
        qqq_map[[
            "Close",
            "QQQ_return_20d"
        ]],
        left_on="Date",
        right_index=True,
        how="left"
    )

    regime.rename(
        columns={
            "Close": "QQQ_close"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # IWM
    # --------------------------------------------------------

    iwm_map = iwm.set_index("Date")

    regime = regime.merge(
        iwm_map[[
            "Close",
            "IWM_return_20d"
        ]],
        left_on="Date",
        right_index=True,
        how="left"
    )

    regime.rename(
        columns={
            "Close": "IWM_close"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # DIA
    # --------------------------------------------------------

    dia_map = dia.set_index("Date")

    regime = regime.merge(
        dia_map[[
            "Close",
            "DIA_return_20d"
        ]],
        left_on="Date",
        right_index=True,
        how="left"
    )

    regime.rename(
        columns={
            "Close": "DIA_close"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # VIX
    # --------------------------------------------------------

    vix_map = vix.set_index("Date")

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

    regime.rename(
        columns={
            "Close": "VIX_level"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # Relative strength
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Trend regime
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VIX regime
    # --------------------------------------------------------

    regime["high_vix"] = (
        regime["VIX_level"]
        >
        regime["VIX_level"].rolling(60).median()
    ).astype(int)

    regime["vix_rising"] = (
        regime["VIX_return_5d"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Market agreement
    # --------------------------------------------------------

    regime["market_agreement"] = (
        (regime["SPY_return_20d"] > 0).astype(int)
        +
        (regime["QQQ_return_20d"] > 0).astype(int)
        +
        (regime["IWM_return_20d"] > 0).astype(int)
        +
        (regime["DIA_return_20d"] > 0).astype(int)
    )

    # --------------------------------------------------------
    # Merge features
    # --------------------------------------------------------

    data = data.merge(
        regime,
        on="Date",
        how="outer",
        suffixes=("", "_new")
    )

    # --------------------------------------------------------
    # Remove future targets from LIVE dataset.
    #
    # V24 must never need to know the future.
    # --------------------------------------------------------

    future_columns = [
        c for c in data.columns
        if c.startswith("future_")
    ]

    data = data.drop(
        columns=future_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Remove training-only target columns.
    # --------------------------------------------------------

    data = data.drop(
        columns=[
            "target",
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .sort_values("Date")
        .dropna()
        .reset_index(drop=True)
    )

    features = validate_feature_schema(data, REFERENCE_FEATURE_FILE)
    validate_feature_values(data, features, "Live V28 features")
    comparison = compare_feature_schema(data, REFERENCE_FEATURE_FILE)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved {len(data)} live rows"
    )

    print(
        f"Feature columns: {len(data.columns) - 1}"
    )

    print()
    print("V28 FEATURE INTEGRITY")
    print(f"85 training features: {len(comparison['expected'])}")
    print(f"85 live features:     {len(comparison['live'])}")
    print(f"MISSING FROM LIVE:    {comparison['missing'] or 'none'}")
    print(f"EXTRA IN LIVE:        {comparison['extra'] or 'none'}")
    print(f"ORDER MATCH:          {comparison['order_match']}")

    print(
        f"Date range: "
        f"{data['Date'].min().date()} "
        f"→ "
        f"{data['Date'].max().date()}"
    )

    print()
    print(
        "LATEST LIVE ROW:"
    )

    print(
        data.tail(1).to_string(
            index=False
        )
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
