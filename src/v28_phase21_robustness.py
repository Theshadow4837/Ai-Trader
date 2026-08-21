"""
V28 PHASE 21 — FROZEN MODEL ROBUSTNESS AUDIT

RESEARCH / PAPER-TRADING ANALYSIS ONLY

NO TRAINING
NO MODEL MODIFICATION
NO BROKER
NO REAL ORDERS

Reads Phase 20 outputs and tests whether the reported
performance is robust across:
- individual trades
- years
- winning/losing trade concentration
- bootstrap trade resampling
- removal of best trades
- removal of worst trades
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("data/v28_phase20")
OUTPUT_DIR = Path("data/v28_phase21")

INITIAL_CAPITAL = 10_000.0

N_BOOTSTRAPS = 10_000
RANDOM_SEED = 2026


def load_files():

    required = [
        "summary.csv",
        "daily_equity.csv",
        "trade_ledger.csv",
        "yearly_results.csv",
        "frozen_predictions.csv",
    ]

    for filename in required:

        path = INPUT_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"Missing Phase 20 file:\n{path}"
            )

    summary = pd.read_csv(
        INPUT_DIR / "summary.csv"
    )

    daily = pd.read_csv(
        INPUT_DIR / "daily_equity.csv"
    )

    trades = pd.read_csv(
        INPUT_DIR / "trade_ledger.csv"
    )

    yearly = pd.read_csv(
        INPUT_DIR / "yearly_results.csv"
    )

    predictions = pd.read_csv(
        INPUT_DIR / "frozen_predictions.csv"
    )

    return (
        summary,
        daily,
        trades,
        yearly,
        predictions,
    )


def extract_completed_trades(trades):

    if len(trades) == 0:
        return pd.DataFrame()

    if "side" not in trades.columns:
        raise RuntimeError(
            "trade_ledger.csv has no 'side' column."
        )

    completed = trades[
        trades["side"].astype(str).str.upper()
        == "SELL"
    ].copy()

    if (
        "completed_trade_return"
        not in completed.columns
    ):

        raise RuntimeError(
            "trade_ledger.csv has no "
            "'completed_trade_return' column."
        )

    completed[
        "completed_trade_return"
    ] = pd.to_numeric(
        completed[
            "completed_trade_return"
        ],
        errors="coerce",
    )

    completed = completed.dropna(
        subset=[
            "completed_trade_return"
        ]
    ).reset_index(
        drop=True
    )

    return completed


def compound_returns(returns):

    values = np.asarray(
        returns,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    return float(
        np.prod(
            1.0 + values
        ) - 1.0
    )


def max_drawdown_from_returns(returns):

    values = np.asarray(
        returns,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    equity = np.cumprod(
        1.0 + values
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity / peaks
    ) - 1.0

    return float(
        drawdowns.min()
    )


def percentile_report(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "p01": np.percentile(
            values,
            1,
        ),
        "p05": np.percentile(
            values,
            5,
        ),
        "p25": np.percentile(
            values,
            25,
        ),
        "p50": np.percentile(
            values,
            50,
        ),
        "p75": np.percentile(
            values,
            75,
        ),
        "p95": np.percentile(
            values,
            95,
        ),
        "p99": np.percentile(
            values,
            99,
        ),
    }


def bootstrap_trade_returns(
    trade_returns,
    rng,
):

    n = len(trade_returns)

    if n == 0:
        return None

    bootstrap_returns = np.empty(
        N_BOOTSTRAPS,
        dtype=float,
    )

    bootstrap_drawdowns = np.empty(
        N_BOOTSTRAPS,
        dtype=float,
    )

    for i in range(
        N_BOOTSTRAPS
    ):

        sample = rng.choice(
            trade_returns,
            size=n,
            replace=True,
        )

        bootstrap_returns[i] = (
            compound_returns(
                sample
            )
        )

        bootstrap_drawdowns[i] = (
            max_drawdown_from_returns(
                sample
            )
        )

    return (
        bootstrap_returns,
        bootstrap_drawdowns,
    )


def concentration_analysis(
    trade_returns,
):

    ordered = np.sort(
        trade_returns
    )[::-1]

    total_return = compound_returns(
        trade_returns
    )

    results = []

    removal_counts = [
        1,
        3,
        5,
        10,
    ]

    for count in removal_counts:

        if count >= len(ordered):
            continue

        remaining = ordered[count:]

        results.append({

            "removed_best_trades":
                count,

            "remaining_trades":
                len(remaining),

            "return":
                compound_returns(
                    remaining
                ),

        })

    # ----------------------------------------------------
    # Remove the worst trades.
    # ----------------------------------------------------

    ordered_worst = np.sort(
        trade_returns
    )

    for count in removal_counts:

        if count >= len(ordered_worst):
            continue

        remaining = ordered_worst[count:]

        results.append({

            "removed_worst_trades":
                count,

            "remaining_trades":
                len(remaining),

            "return":
                compound_returns(
                    remaining
                ),

        })

    return pd.DataFrame(
        results
    )


def yearly_consistency(
    yearly,
):

    if len(yearly) == 0:
        return {}

    returns = pd.to_numeric(
        yearly["return"],
        errors="coerce",
    ).dropna()

    return {

        "years":
            len(returns),

        "positive_years":
            int(
                (returns > 0).sum()
            ),

        "negative_years":
            int(
                (returns < 0).sum()
            ),

        "median_year_return":
            float(
                returns.median()
            ),

        "worst_year":
            float(
                returns.min()
            ),

        "best_year":
            float(
                returns.max()
            ),

    }


def main():

    print()
    print("=" * 70)
    print("V28 PHASE 21 — ROBUSTNESS AUDIT")
    print("=" * 70)

    print()
    print("RESEARCH / PAPER-TRADING ANALYSIS ONLY")
    print("NO TRAINING")
    print("NO MODEL MODIFICATION")
    print("NO BROKER")
    print("NO REAL ORDERS")

    (
        summary,
        daily,
        trades,
        yearly,
        predictions,
    ) = load_files()

    completed = extract_completed_trades(
        trades
    )

    if len(completed) < 10:

        raise RuntimeError(
            "Not enough completed trades "
            "for robustness analysis."
        )

    trade_returns = (
        completed[
            "completed_trade_return"
        ]
        .astype(float)
        .to_numpy()
    )

    # ----------------------------------------------------
    # Basic trade statistics.
    # ----------------------------------------------------

    winners = (
        trade_returns > 0
    )

    losers = (
        trade_returns < 0
    )

    gross_winner_return = (
        trade_returns[winners].sum()
        if winners.any()
        else 0.0
    )

    gross_loser_return = (
        np.abs(
            trade_returns[losers]
        ).sum()
        if losers.any()
        else 0.0
    )

    profit_factor = (
        gross_winner_return
        / gross_loser_return
        if gross_loser_return > 0
        else np.inf
    )

    # ----------------------------------------------------
    # Top trade concentration.
    # ----------------------------------------------------

    sorted_returns = np.sort(
        trade_returns
    )[::-1]

    top_1 = (
        sorted_returns[:1].sum()
    )

    top_5 = (
        sorted_returns[:5].sum()
    )

    top_10 = (
        sorted_returns[:10].sum()
    )

    # ----------------------------------------------------
    # Bootstrap.
    # ----------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap = bootstrap_trade_returns(
        trade_returns,
        rng,
    )

    (
        bootstrap_returns,
        bootstrap_drawdowns,
    ) = bootstrap

    return_report = percentile_report(
        bootstrap_returns
    )

    dd_report = percentile_report(
        bootstrap_drawdowns
    )

    # ----------------------------------------------------
    # Best-trade removal.
    # ----------------------------------------------------

    concentration = concentration_analysis(
        trade_returns
    )

    # ----------------------------------------------------
    # Year consistency.
    # ----------------------------------------------------

    consistency = yearly_consistency(
        yearly
    )

    # ----------------------------------------------------
    # Print results.
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print("TRADE DISTRIBUTION")
    print("=" * 70)

    print(
        f"Completed trades: "
        f"{len(trade_returns)}"
    )

    print(
        f"Winners: "
        f"{int(winners.sum())}"
    )

    print(
        f"Losers: "
        f"{int(losers.sum())}"
    )

    print(
        f"Win rate: "
        f"{winners.mean() * 100:.2f}%"
    )

    print(
        f"Profit factor: "
        f"{profit_factor:.3f}"
    )

    print(
        f"Mean trade: "
        f"{trade_returns.mean() * 100:+.3f}%"
    )

    print(
        f"Median trade: "
        f"{np.median(trade_returns) * 100:+.3f}%"
    )

    print()
    print(
        f"Best trade: "
        f"{trade_returns.max() * 100:+.2f}%"
    )

    print(
        f"Worst trade: "
        f"{trade_returns.min() * 100:+.2f}%"
    )

    print()
    print("=" * 70)
    print("RETURN CONCENTRATION")
    print("=" * 70)

    print(
        f"Top 1 trade simple return contribution: "
        f"{top_1 * 100:+.2f}%"
    )

    print(
        f"Top 5 trades simple return contribution: "
        f"{top_5 * 100:+.2f}%"
    )

    print(
        f"Top 10 trades simple return contribution: "
        f"{top_10 * 100:+.2f}%"
    )

    print()
    print("=" * 70)
    print("BOOTSTRAP ROBUSTNESS")
    print("=" * 70)

    print(
        f"Bootstrap samples: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"Return P01: "
        f"{return_report['p01'] * 100:+.2f}%"
    )

    print(
        f"Return P05: "
        f"{return_report['p05'] * 100:+.2f}%"
    )

    print(
        f"Return P25: "
        f"{return_report['p25'] * 100:+.2f}%"
    )

    print(
        f"Return P50: "
        f"{return_report['p50'] * 100:+.2f}%"
    )

    print(
        f"Return P75: "
        f"{return_report['p75'] * 100:+.2f}%"
    )

    print(
        f"Return P95: "
        f"{return_report['p95'] * 100:+.2f}%"
    )

    print(
        f"Return P99: "
        f"{return_report['p99'] * 100:+.2f}%"
    )

    print()
    print(
        f"Bootstrap probability of loss: "
        f"{(
            bootstrap_returns < 0
        ).mean() * 100:.2f}%"
    )

    print()
    print("=" * 70)
    print("BEST-TRADE REMOVAL TEST")
    print("=" * 70)

    print(
        concentration.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("YEAR CONSISTENCY")
    print("=" * 70)

    print(
        f"Years: "
        f"{consistency['years']}"
    )

    print(
        f"Positive years: "
        f"{consistency['positive_years']}"
    )

    print(
        f"Negative years: "
        f"{consistency['negative_years']}"
    )

    print(
        f"Median yearly return: "
        f"{consistency['median_year_return'] * 100:+.2f}%"
    )

    print(
        f"Worst year: "
        f"{consistency['worst_year'] * 100:+.2f}%"
    )

    print(
        f"Best year: "
        f"{consistency['best_year'] * 100:+.2f}%"
    )

    # ----------------------------------------------------
    # Save outputs.
    # ----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    concentration.to_csv(
        OUTPUT_DIR
        / "concentration_analysis.csv",
        index=False,
    )

    bootstrap_df = pd.DataFrame({

        "bootstrap_return":
            bootstrap_returns,

        "bootstrap_max_drawdown":
            bootstrap_drawdowns,

    })

    bootstrap_df.to_csv(
        OUTPUT_DIR
        / "bootstrap_results.csv",
        index=False,
    )

    trade_distribution = pd.DataFrame({

        "trade_return":
            trade_returns,

    })

    trade_distribution.to_csv(
        OUTPUT_DIR
        / "trade_distribution.csv",
        index=False,
    )

    audit_summary = {

        "completed_trades":
            len(trade_returns),

        "win_rate":
            winners.mean(),

        "profit_factor":
            profit_factor,

        "mean_trade":
            trade_returns.mean(),

        "median_trade":
            np.median(
                trade_returns
            ),

        "best_trade":
            trade_returns.max(),

        "worst_trade":
            trade_returns.min(),

        "bootstrap_return_p01":
            return_report["p01"],

        "bootstrap_return_p05":
            return_report["p05"],

        "bootstrap_return_p25":
            return_report["p25"],

        "bootstrap_return_median":
            return_report["p50"],

        "bootstrap_return_p75":
            return_report["p75"],

        "bootstrap_return_p95":
            return_report["p95"],

        "bootstrap_return_p99":
            return_report["p99"],

        "bootstrap_loss_probability":
            (
                bootstrap_returns < 0
            ).mean(),

        "bootstrap_dd_p05":
            dd_report["p05"],

        "bootstrap_dd_median":
            dd_report["p50"],

        "bootstrap_dd_p95":
            dd_report["p95"],

        "positive_years":
            consistency[
                "positive_years"
            ],

        "negative_years":
            consistency[
                "negative_years"
            ],

    }

    pd.DataFrame(
        [audit_summary]
    ).to_csv(
        OUTPUT_DIR
        / "audit_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("PHASE 21 OUTPUTS")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_DIR}/"
    )

    print()
    print("Files:")
    print(
        "    audit_summary.csv"
    )
    print(
        "    bootstrap_results.csv"
    )
    print(
        "    concentration_analysis.csv"
    )
    print(
        "    trade_distribution.csv"
    )

    print()
    print("=" * 70)
    print("V28 STATUS")
    print("=" * 70)

    print("    FROZEN")
    print("    NOT TRAINED")
    print("    NOT MODIFIED")
    print("    NO BROKER")
    print("    NO REAL ORDERS")

    print()
    print("=" * 70)
    print("PHASE 21 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
