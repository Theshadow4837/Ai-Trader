# ============================================================
# V28 PAPER PERFORMANCE ANALYZER
# ============================================================
#
# READ-ONLY ANALYSIS
# Does NOT:
#   - train the model
#   - modify the frozen model
#   - place real orders
#   - modify paper-account files
#
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

LOG_FILE = Path("data/v28_paper_log.csv")
EQUITY_FILE = Path("data/v28_paper_equity.csv")
TRADES_FILE = Path("data/v28_paper_trades.csv")
SPY_FILE = Path("data/SPY.csv")

TRADING_DAYS = 252


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, required=True):
    """
    Find a column using case-insensitive matching.
    """

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    if required:
        raise RuntimeError(
            f"Could not find required column.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def pct(x):
    if pd.isna(x):
        return "N/A"

    return f"{x * 100:.2f}%"


def money(x):
    if pd.isna(x):
        return "N/A"

    return f"${x:,.2f}"


# ============================================================
# LOAD EQUITY
# ============================================================

def load_equity():

    if not EQUITY_FILE.exists():
        raise RuntimeError(
            f"Missing equity file:\n{EQUITY_FILE}"
        )

    df = pd.read_csv(EQUITY_FILE)

    date_col = find_column(
        df,
        ["date", "Date", "timestamp", "Timestamp"]
    )

    equity_col = find_column(
        df,
        [
            "equity",
            "current_equity",
            "account_equity",
            "portfolio_equity",
            "total_equity",
        ]
    )

    df["Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["Equity"] = pd.to_numeric(
        df[equity_col],
        errors="coerce"
    )

    df = (
        df[
            ["Date", "Equity"]
        ]
        .dropna()
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        raise RuntimeError(
            "Not enough equity observations."
        )

    return df


# ============================================================
# LOAD TRADES
# ============================================================

def load_trades():

    if not TRADES_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(TRADES_FILE)

    if len(df) == 0:
        return df

    return df


# ============================================================
# LOAD DECISION LOG
# ============================================================

def load_log():

    if not LOG_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(LOG_FILE)

    if len(df) == 0:
        return df

    return df


# ============================================================
# LOAD ACCOUNT
# ============================================================

def load_account():

    account_file = Path(
        "data/v28_paper_account.json"
    )

    if not account_file.exists():
        return {}

    with open(account_file, "r") as f:
        return json.load(f)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(equity):

    equity = equity.copy()

    equity["Return"] = (
        equity["Equity"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    returns = (
        equity["Return"]
        .dropna()
    )

    starting_equity = float(
        equity["Equity"].iloc[0]
    )

    ending_equity = float(
        equity["Equity"].iloc[-1]
    )

    total_return = (
        ending_equity /
        starting_equity
        - 1.0
    )

    days = (
        equity["Date"].iloc[-1]
        - equity["Date"].iloc[0]
    ).days

    years = days / 365.25

    if years > 0:
        cagr = (
            ending_equity /
            starting_equity
        ) ** (1 / years) - 1
    else:
        cagr = np.nan

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    rolling_peak = (
        equity["Equity"]
        .cummax()
    )

    drawdown = (
        equity["Equity"] /
        rolling_peak
        - 1.0
    )

    max_drawdown = drawdown.min()

    # Drawdown duration
    underwater = drawdown < 0

    max_duration = 0
    current_duration = 0

    for value in underwater:

        if value:
            current_duration += 1
            max_duration = max(
                max_duration,
                current_duration
            )
        else:
            current_duration = 0

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    if (
        len(returns) > 1
        and returns.std(ddof=1) > 0
    ):
        sharpe = (
            np.sqrt(TRADING_DAYS)
            * returns.mean()
            / returns.std(ddof=1)
        )
    else:
        sharpe = np.nan

    # --------------------------------------------------------
    # Sortino
    # --------------------------------------------------------

    downside = returns[
        returns < 0
    ]

    if (
        len(downside) > 0
        and downside.std(ddof=1) > 0
    ):
        sortino = (
            np.sqrt(TRADING_DAYS)
            * returns.mean()
            / downside.std(ddof=1)
        )
    else:
        sortino = np.nan

    return {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_duration,
        "sharpe": sharpe,
        "sortino": sortino,
        "observations": len(equity),
        "days": days,
    }


# ============================================================
# SPY BENCHMARK
# ============================================================

def calculate_spy():

    if not SPY_FILE.exists():
        return None

    spy = pd.read_csv(SPY_FILE)

    date_col = find_column(
        spy,
        ["date", "Date"]
    )

    close_col = find_column(
        spy,
        ["close", "Close"]
    )

    spy["Date"] = pd.to_datetime(
        spy[date_col],
        errors="coerce"
    )

    spy["Close"] = pd.to_numeric(
        spy[close_col],
        errors="coerce"
    )

    spy = (
        spy[
            ["Date", "Close"]
        ]
        .dropna()
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    if len(spy) < 2:
        return None

    start_price = spy["Close"].iloc[0]
    end_price = spy["Close"].iloc[-1]

    total_return = (
        end_price /
        start_price
        - 1.0
    )

    days = (
        spy["Date"].iloc[-1]
        - spy["Date"].iloc[0]
    ).days

    years = days / 365.25

    if years > 0:
        cagr = (
            end_price /
            start_price
        ) ** (1 / years) - 1
    else:
        cagr = np.nan

    spy["Return"] = (
        spy["Close"]
        .pct_change()
    )

    rolling_peak = (
        spy["Close"]
        .cummax()
    )

    drawdown = (
        spy["Close"] /
        rolling_peak
        - 1.0
    )

    max_drawdown = drawdown.min()

    return {
        "start_date": spy["Date"].iloc[0],
        "end_date": spy["Date"].iloc[-1],
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# TRADE STATISTICS
# ============================================================

def calculate_trade_stats(trades):

    if trades.empty:
        return {
            "trade_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": np.nan,
            "average_trade": np.nan,
            "profit_factor": np.nan,
            "transaction_costs": np.nan,
        }

    pnl_col = find_column(
        trades,
        [
            "pnl",
            "realized_pnl",
            "profit",
            "trade_pnl",
        ],
        required=False,
    )

    cost_col = find_column(
        trades,
        [
            "transaction_cost",
            "transaction_costs",
            "cost",
            "fees",
        ],
        required=False,
    )

    if pnl_col is not None:

        pnl = pd.to_numeric(
            trades[pnl_col],
            errors="coerce"
        ).dropna()

        winning = pnl[pnl > 0]
        losing = pnl[pnl < 0]

        trade_count = len(pnl)

        winning_trades = len(winning)
        losing_trades = len(losing)

        win_rate = (
            winning_trades / trade_count
            if trade_count > 0
            else np.nan
        )

        average_trade = (
            pnl.mean()
            if len(pnl)
            else np.nan
        )

        gross_profit = (
            winning.sum()
        )

        gross_loss = abs(
            losing.sum()
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit /
                gross_loss
            )
        else:
            profit_factor = np.inf

    else:

        trade_count = len(trades)
        winning_trades = np.nan
        losing_trades = np.nan
        win_rate = np.nan
        average_trade = np.nan
        profit_factor = np.nan

    if cost_col is not None:

        costs = pd.to_numeric(
            trades[cost_col],
            errors="coerce"
        )

        transaction_costs = (
            costs.sum()
        )

    else:
        transaction_costs = np.nan

    return {
        "trade_count": trade_count,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "average_trade": average_trade,
        "profit_factor": profit_factor,
        "transaction_costs": transaction_costs,
    }


# ============================================================
# YEARLY PERFORMANCE
# ============================================================

def yearly_performance(equity):

    df = equity.copy()

    df["Year"] = (
        df["Date"]
        .dt.year
    )

    rows = []

    for year, group in df.groupby("Year"):

        start = group["Equity"].iloc[0]
        end = group["Equity"].iloc[-1]

        yearly_return = (
            end / start - 1
        )

        peak = (
            group["Equity"]
            .cummax()
        )

        dd = (
            group["Equity"] /
            peak
            - 1
        )

        rows.append({
            "Year": int(year),
            "Return": yearly_return,
            "Max Drawdown": dd.min(),
        })

    return pd.DataFrame(rows)


# ============================================================
# FORWARD PERIOD
# ============================================================
#
# The current paper run has historical observations from
# 2015 onward. We report the entire curve above, but we also
# show the most recent 30/90/180-day periods separately.
#
# This avoids pretending the entire historical curve is
# forward performance.
# ============================================================

def recent_period_metrics(equity, days):

    latest_date = equity["Date"].max()

    cutoff = (
        latest_date
        - pd.Timedelta(days=days)
    )

    period = equity[
        equity["Date"] >= cutoff
    ].copy()

    if len(period) < 2:
        return None

    return calculate_metrics(period)


# ============================================================
# PRINT
# ============================================================

def print_report(
    metrics,
    trade_stats,
    spy,
    yearly,
    account,
    equity,
):

    print()
    print("=" * 60)
    print("V28 PAPER PERFORMANCE")
    print("=" * 60)

    print()
    print("ACCOUNT")
    print("-" * 60)

    print(
        f"Starting equity:       "
        f"{money(metrics['starting_equity'])}"
    )

    print(
        f"Ending equity:         "
        f"{money(metrics['ending_equity'])}"
    )

    print(
        f"Total return:          "
        f"{pct(metrics['total_return'])}"
    )

    print(
        f"Annualized return:     "
        f"{pct(metrics['cagr'])}"
    )

    print(
        f"Observations:          "
        f"{metrics['observations']}"
    )

    print(
        f"Days:                  "
        f"{metrics['days']}"
    )

    print()
    print("RISK")
    print("-" * 60)

    print(
        f"Sharpe ratio:          "
        f"{metrics['sharpe']:.3f}"
        if not pd.isna(metrics["sharpe"])
        else "Sharpe ratio:          N/A"
    )

    print(
        f"Sortino ratio:         "
        f"{metrics['sortino']:.3f}"
        if not pd.isna(metrics["sortino"])
        else "Sortino ratio:         N/A"
    )

    print(
        f"Maximum drawdown:      "
        f"{pct(metrics['max_drawdown'])}"
    )

    print(
        f"Max drawdown duration: "
        f"{metrics['max_drawdown_duration']} observations"
    )

    print()
    print("TRADING")
    print("-" * 60)

    print(
        f"Trades:                "
        f"{trade_stats['trade_count']}"
    )

    print(
        f"Winning trades:        "
        f"{trade_stats['winning_trades']}"
    )

    print(
        f"Losing trades:         "
        f"{trade_stats['losing_trades']}"
    )

    print(
        f"Win rate:              "
        f"{pct(trade_stats['win_rate'])}"
    )

    print(
        f"Average trade:         "
        f"{money(trade_stats['average_trade'])}"
    )

    print(
        f"Profit factor:         "
        f"{trade_stats['profit_factor']:.3f}"
        if not pd.isna(trade_stats["profit_factor"])
        else "Profit factor:         N/A"
    )

    print(
        f"Transaction costs:     "
        f"{money(trade_stats['transaction_costs'])}"
    )

    # --------------------------------------------------------
    # Account JSON
    # --------------------------------------------------------

    if account:

        print()
        print("ACCOUNT FILE")
        print("-" * 60)

        for key in [
            "starting_capital",
            "cash",
            "current_equity",
            "realized_pnl",
            "unrealized_pnl",
            "cumulative_return",
            "peak_equity",
            "maximum_drawdown",
            "trade_count",
            "days_in_market",
        ]:

            if key in account:

                value = account[key]

                if (
                    "capital" in key
                    or key in [
                        "cash",
                        "current_equity",
                        "realized_pnl",
                        "unrealized_pnl",
                        "peak_equity",
                    ]
                ):
                    value = money(value)

                elif key == "cumulative_return":
                    value = pct(value)

                print(
                    f"{key:24s} {value}"
                )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print()
    print("SPY BUY-AND-HOLD")
    print("-" * 60)

    if spy is not None:

        print(
            f"SPY period:            "
            f"{spy['start_date'].date()} → "
            f"{spy['end_date'].date()}"
        )

        print(
            f"SPY return:            "
            f"{pct(spy['total_return'])}"
        )

        print(
            f"SPY annualized return: "
            f"{pct(spy['cagr'])}"
        )

        print(
            f"SPY max drawdown:      "
            f"{pct(spy['max_drawdown'])}"
        )

        print(
            f"V28 minus SPY:         "
            f"{pct(metrics['total_return'] - spy['total_return'])}"
        )

    else:

        print("SPY data unavailable.")

    # --------------------------------------------------------
    # Recent periods
    # --------------------------------------------------------

    print()
    print("RECENT PERFORMANCE")
    print("-" * 60)

    for days in [30, 90, 180]:

        result = recent_period_metrics(
            equity,
            days
        )

        if result is None:
            continue

        print()
        print(f"Last {days} days:")

        print(
            f"  Return:       "
            f"{pct(result['total_return'])}"
        )

        print(
            f"  Sharpe:       "
            f"{result['sharpe']:.3f}"
            if not pd.isna(result["sharpe"])
            else "  Sharpe:       N/A"
        )

        print(
            f"  Max drawdown: "
            f"{pct(result['max_drawdown'])}"
        )

    # --------------------------------------------------------
    # Yearly
    # --------------------------------------------------------

    print()
    print("YEARLY PERFORMANCE")
    print("-" * 60)

    if len(yearly):

        for _, row in yearly.iterrows():

            print(
                f"{int(row['Year'])}: "
                f"return={pct(row['Return'])} "
                f"drawdown={pct(row['Max Drawdown'])}"
            )

    print()
    print("=" * 60)
    print("V28 PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print("MODEL STATUS:")
    print("    FROZEN")

    print("TRAINING:")
    print("    NONE")

    print("REAL ORDERS:")
    print("    NONE")

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V28 PAPER PERFORMANCE ANALYZER")
    print("=" * 60)

    print()
    print("READ-ONLY ANALYSIS")
    print("Frozen model will NOT be modified.")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    equity = load_equity()
    trades = load_trades()
    account = load_account()
    load_log()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        equity
    )

    trade_stats = calculate_trade_stats(
        trades
    )

    spy = calculate_spy()

    yearly = yearly_performance(
        equity
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        metrics,
        trade_stats,
        spy,
        yearly,
        account,
        equity,
    )


if __name__ == "__main__":
    main()
