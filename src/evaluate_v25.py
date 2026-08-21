"""
============================================================
V25 HOLDOUT EVALUATOR
============================================================

Research / paper trading ONLY.

Training data:
    2015 -> 2023

Evaluation data:
    2024 -> latest available

IMPORTANT:
    The model is NOT retrained during evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from trading_env_v25 import TradingEnvironment


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path("models/v25_ppo.zip")

DATA_FILE = Path(
    "data/market_features_v14.csv"
)

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005

TEST_START = pd.Timestamp(
    "2024-01-01"
)


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(equity):

    equity = pd.Series(
        equity,
        dtype=float
    )

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1.0
    )

    return float(
        drawdown.min()
    )


# ============================================================
# SHARPE
# ============================================================

def sharpe_ratio(returns):

    returns = pd.Series(
        returns,
        dtype=float
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return float(
        np.sqrt(252)
        * returns.mean()
        / std
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(
        f"[V25] Loading model: "
        f"{MODEL_PATH}"
    )

    model = PPO.load(
        str(MODEL_PATH)
    )

    return model


# ============================================================
# PREPARE TEST DATA
# ============================================================

def load_test_data():

    data = pd.read_csv(
        DATA_FILE
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .reset_index(drop=True)
    )

    test = data[
        data["Date"] >= TEST_START
    ].copy()

    if len(test) < 10:

        raise ValueError(
            "Not enough test data."
        )

    return test


# ============================================================
# BUILD TEST ENVIRONMENT
# ============================================================

def build_test_environment():

    env = TradingEnvironment(
        data_file=DATA_FILE,
        start_date="2024-01-01",
        end_date="2099-12-31",
        episode_length=10_000,
        transaction_cost=TRANSACTION_COST,
        initial_capital=INITIAL_CAPITAL,
        seed=42,
    )

    return env


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print()
    print("=" * 60)
    print("V25 HOLDOUT EVALUATION")
    print("=" * 60)

    model = load_model()

    env = build_test_environment()

    print()
    print(
        f"[V25] Test rows: "
        f"{len(env.data)}"
    )

    print(
        f"[V25] Test period: "
        f"{env.data['Date'].min().date()} "
        f"→ "
        f"{env.data['Date'].max().date()}"
    )

    print(
        f"[V25] Features: "
        f"{len(env.features)}"
    )

    env.leakage_check()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=42
    )

    # Force evaluation to start at the first test row.
    env.current_step = 0
    env.episode_start = 0
    env.episode_end = len(env.X) - 1

    observation = env.X[
        env.current_step
    ].copy()

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    equity = [
        INITIAL_CAPITAL
    ]

    daily_returns = []

    actions = []

    dates = []

    trade_count = 0

    previous_position = 0

    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    while True:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        action = int(action)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        actions.append(
            action
        )

        dates.append(
            info["previous_date"]
        )

        current_equity = float(
            info["equity"]
        )

        equity.append(
            current_equity
        )

        strategy_return = float(
            info["strategy_return"]
        )

        daily_returns.append(
            strategy_return
        )

        if action != previous_position:

            trade_count += 1

        previous_position = (
            action
        )

        if terminated or truncated:

            break

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    equity_series = pd.Series(
        equity
    )

    final_value = float(
        equity_series.iloc[-1]
    )

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1.0
    )

    years = (
        (
            env.data["Date"].iloc[-1]
            - env.data["Date"].iloc[0]
        ).days
        / 365.25
    )

    if years > 0:

        annualized = (
            (
                final_value
                / INITIAL_CAPITAL
            )
            ** (1.0 / years)
            - 1.0
        )

    else:

        annualized = 0.0

    mdd = max_drawdown(
        equity_series
    )

    sharpe = sharpe_ratio(
        daily_returns
    )

    # --------------------------------------------------------
    # ACTION STATISTICS
    # --------------------------------------------------------

    action_series = pd.Series(
        actions
    )

    hold_count = int(
        (action_series == 0).sum()
    )

    buy_count = int(
        (action_series == 1).sum()
    )

    sell_count = int(
        (action_series == 2).sum()
    )

    # --------------------------------------------------------
    # WIN RATE
    # --------------------------------------------------------

    positive = sum(
        r > 0
        for r in daily_returns
    )

    negative = sum(
        r < 0
        for r in daily_returns
    )

    active_returns = [
        r
        for r in daily_returns
        if abs(r) > 1e-12
    ]

    if active_returns:

        win_rate = (
            sum(
                r > 0
                for r in active_returns
            )
            / len(active_returns)
        )

        average_trade = float(
            np.mean(
                active_returns
            )
        )

    else:

        win_rate = 0.0
        average_trade = 0.0

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V25 HOLDOUT RESULTS")
    print("=" * 60)

    print(
        f"Final value: "
        f"${final_value:,.2f}"
    )

    print(
        f"Return: "
        f"{total_return:.2%}"
    )

    print(
        f"Annualized: "
        f"{annualized:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{mdd:.2%}"
    )

    print(
        f"Sharpe: "
        f"{sharpe:.3f}"
    )

    print(
        f"Trades: "
        f"{trade_count}"
    )

    print(
        f"Win rate: "
        f"{win_rate:.2%}"
    )

    print(
        f"Average active return: "
        f"{average_trade:.4%}"
    )

    print()
    print("=" * 60)
    print("ACTION DISTRIBUTION")
    print("=" * 60)

    print(
        f"HOLD: "
        f"{hold_count}"
    )

    print(
        f"BUY:  "
        f"{buy_count}"
    )

    print(
        f"SELL: "
        f"{sell_count}"
    )

    total_actions = len(
        action_series
    )

    if total_actions:

        print()
        print(
            f"HOLD: "
            f"{hold_count / total_actions:.2%}"
        )

        print(
            f"BUY:  "
            f"{buy_count / total_actions:.2%}"
        )

        print(
            f"SELL: "
            f"{sell_count / total_actions:.2%}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame({

        "Date":
            dates,

        "action":
            actions,

        "strategy_return":
            daily_returns,

        "equity":
            equity_series.iloc[
                1:
            ].values,
    })

    output = Path(
        "data/v25_holdout_results.csv"
    )

    results.to_csv(
        output,
        index=False
    )

    print()
    print(
        f"[V25] Results saved:"
    )

    print(
        f"       {output}"
    )

    env.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()