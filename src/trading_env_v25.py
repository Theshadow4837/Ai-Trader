"""
============================================================
V25 TRADING ENVIRONMENT
============================================================

Research / paper-trading environment only.

This is the environment layer for the future automated
learning system.

Actions:
    0 = HOLD
    1 = BUY
    2 = SELL

Important:
    - No broker connection.
    - No real-money trading.
    - Training data ends at 2023-12-29.
    - 2024+ remains held out.
    - Future information is never included in observations.
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("data/market_features_v14.csv")

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-29"

INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.0005

DEFAULT_EPISODE_LENGTH = 252


# ============================================================
# FORBIDDEN OBSERVATION TERMS
# ============================================================

FORBIDDEN_FEATURE_WORDS = {
    "future",
    "target",
    "label",
    "reward",
}


# ============================================================
# ACTIONS
# ============================================================

HOLD = 0
BUY = 1
SELL = 2


# ============================================================
# ENVIRONMENT
# ============================================================

class TradingEnvironment(gym.Env):

    metadata = {
        "render_modes": ["human"],
    }

    def __init__(
        self,
        data_file=DATA_FILE,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        episode_length=DEFAULT_EPISODE_LENGTH,
        transaction_cost=TRANSACTION_COST,
        initial_capital=INITIAL_CAPITAL,
        seed=None,
    ):

        super().__init__()

        self.data_file = Path(data_file)

        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        self.episode_length = int(
            episode_length
        )

        self.transaction_cost = float(
            transaction_cost
        )

        self.initial_capital = float(
            initial_capital
        )

        self.rng = np.random.default_rng(seed)

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        self.data = pd.read_csv(
            self.data_file
        )

        if "Date" not in self.data.columns:
            raise ValueError(
                "Dataset must contain a Date column."
            )

        self.data["Date"] = pd.to_datetime(
            self.data["Date"]
        )

        self.data = (
            self.data
            .sort_values("Date")
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # SELECT TRAINING PERIOD FIRST
        #
        # This is important because reward arrays and feature
        # arrays must remain perfectly aligned.
        # ----------------------------------------------------

        mask = (
            (self.data["Date"] >= self.start_date)
            &
            (self.data["Date"] <= self.end_date)
        )

        self.data = (
            self.data.loc[mask]
            .reset_index(drop=True)
        )

        if len(self.data) < 100:
            raise ValueError(
                "Not enough training data."
            )

        # ----------------------------------------------------
        # FIND FEATURES
        # ----------------------------------------------------

        self.features = self._find_features()

        if not self.features:
            raise ValueError(
                "No usable features were found."
            )

        # ----------------------------------------------------
        # REMOVE ROWS WITH MISSING FEATURES
        # ----------------------------------------------------

        self.data = (
            self.data
            .dropna(
                subset=self.features
            )
            .reset_index(drop=True)
        )

        if len(self.data) < 100:
            raise ValueError(
                "Not enough valid training rows after "
                "removing missing feature values."
            )

        # ----------------------------------------------------
        # CREATE INTERNAL REWARD RETURN
        #
        # The agent does NOT receive this value.
        #
        # At timestep t:
        #
        #   observation(t)
        #        ↓
        #   choose action
        #        ↓
        #   next market day
        #        ↓
        #   reward
        #
        # We use the next day's SPY return as the simulated
        # one-day market outcome.
        # ----------------------------------------------------

        if "future_1d_return" in self.data.columns:

            self.reward_returns = (
                self.data["future_1d_return"]
                .astype(np.float32)
                .to_numpy()
            )

        elif "SPY_return_1d" in self.data.columns:

            self.reward_returns = (
                self.data["SPY_return_1d"]
                .shift(-1)
                .astype(np.float32)
                .to_numpy()
                .copy()
            )

            # Last row has no next-day return.
            self.reward_returns[-1] = 0.0

        else:

            raise ValueError(
                "Dataset contains neither "
                "future_1d_return nor SPY_return_1d."
            )

        # ----------------------------------------------------
        # OBSERVATION MATRIX
        # ----------------------------------------------------

        self.X = (
            self.data[self.features]
            .astype(np.float32)
            .to_numpy()
        )

        # ----------------------------------------------------
        # NORMALIZATION
        #
        # These statistics are calculated ONLY from the
        # training period.
        # ----------------------------------------------------

        self.mean = np.nanmean(
            self.X,
            axis=0
        )

        self.std = np.nanstd(
            self.X,
            axis=0
        )

        self.std[
            self.std < 1e-8
        ] = 1.0

        self.X = (
            self.X - self.mean
        ) / self.std

        self.X = np.nan_to_num(
            self.X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.X = np.clip(
            self.X,
            -10.0,
            10.0
        ).astype(np.float32)

        # ----------------------------------------------------
        # GYM SPACES
        # ----------------------------------------------------

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.features),),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.current_step = 0
        self.episode_start = 0
        self.episode_end = 0

        self.position = HOLD

        self.capital = (
            self.initial_capital
        )

        self.equity = (
            self.initial_capital
        )

        self.previous_equity = (
            self.initial_capital
        )

        self.trade_count = 0
        self.total_reward = 0.0

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    def _find_features(self):

        excluded = {
            "Date",
            "future_1d_return",
            "future_3d_return",
            "future_5d_return",
            "future_10d_return",
            "target",
            "trade_reward",
            "trade_label",
        }

        features = []

        for column in self.data.columns:

            if column in excluded:
                continue

            lower = column.lower()

            dangerous = any(
                word in lower
                for word in FORBIDDEN_FEATURE_WORDS
            )

            if dangerous:
                continue

            if lower in {
                "date",
                "datetime",
                "timestamp",
            }:
                continue

            features.append(column)

        return features

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    def leakage_check(self):

        leaks = []

        for feature in self.features:

            lower = feature.lower()

            for word in FORBIDDEN_FEATURE_WORDS:

                if word in lower:

                    leaks.append(feature)

                    break

        if leaks:

            raise RuntimeError(
                "LEAKAGE DETECTED:\n"
                + "\n".join(leaks)
            )

        print(
            "[ENV] Leakage check: PASSED"
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        minimum_start = 0

        maximum_start = max(
            0,
            len(self.X)
            - self.episode_length
            - 2
        )

        if maximum_start > minimum_start:

            self.episode_start = int(
                self.np_random.integers(
                    minimum_start,
                    maximum_start + 1
                )
            )

        else:

            self.episode_start = 0

        self.episode_end = min(
            self.episode_start
            + self.episode_length,
            len(self.X) - 1
        )

        self.current_step = (
            self.episode_start
        )

        self.position = HOLD

        self.capital = (
            self.initial_capital
        )

        self.equity = (
            self.initial_capital
        )

        self.previous_equity = (
            self.initial_capital
        )

        self.trade_count = 0
        self.total_reward = 0.0

        observation = (
            self.X[
                self.current_step
            ].copy()
        )

        info = {
            "date":
                self.data[
                    "Date"
                ].iloc[
                    self.current_step
                ],

            "equity":
                self.equity,

            "position":
                self.position,
        }

        return observation, info

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        action = int(action)

        if not self.action_space.contains(action):

            raise ValueError(
                f"Invalid action: {action}"
            )

        current_date = (
            self.data[
                "Date"
            ].iloc[
                self.current_step
            ]
        )

        # ----------------------------------------------------
        # ACTION → POSITION
        # ----------------------------------------------------

        if action == BUY:

            new_position = 1

        elif action == SELL:

            new_position = -1

        else:

            new_position = 0

        # ----------------------------------------------------
        # TRANSACTION COST
        # ----------------------------------------------------

        position_changed = (
            new_position != self.position
        )

        cost = 0.0

        if position_changed:

            cost = self.transaction_cost
            self.trade_count += 1

        # ----------------------------------------------------
        # INTERNAL MARKET RETURN
        # ----------------------------------------------------

        market_return = float(
            self.reward_returns[
                self.current_step
            ]
        )

        # ----------------------------------------------------
        # STRATEGY RETURN
        # ----------------------------------------------------

        if new_position == BUY:

            strategy_return = market_return

        elif new_position == SELL:

            strategy_return = -market_return

        else:

            strategy_return = 0.0

        strategy_return -= cost

        # ----------------------------------------------------
        # UPDATE EQUITY
        # ----------------------------------------------------

        self.previous_equity = (
            self.equity
        )

        self.equity *= (
            1.0 + strategy_return
        )

        self.position = new_position
        self.capital = self.equity

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        if self.previous_equity > 0:

            reward = float(
                np.log(
                    self.equity
                    / self.previous_equity
                )
            )

        else:

            reward = -1.0

        reward = float(
            np.clip(
                reward,
                -1.0,
                1.0
            )
        )

        self.total_reward += reward

        # ----------------------------------------------------
        # ADVANCE
        # ----------------------------------------------------

        next_step = (
            self.current_step + 1
        )

        terminated = (
            next_step >= self.episode_end
        )

        if next_step >= len(self.data):

            terminated = True

            next_step = (
                len(self.data) - 1
            )

        self.current_step = next_step

        observation = (
            self.X[
                self.current_step
            ].copy()
        )

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        info = {

            "date":
                self.data[
                    "Date"
                ].iloc[
                    self.current_step
                ],

            "previous_date":
                current_date,

            "action":
                action,

            "position":
                self.position,

            "market_return":
                market_return,

            "strategy_return":
                strategy_return,

            "transaction_cost":
                cost,

            "equity":
                self.equity,

            "trade_count":
                self.trade_count,

            "total_reward":
                self.total_reward,
        }

        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        date = (
            self.data[
                "Date"
            ].iloc[
                self.current_step
            ]
        )

        print(
            f"[ENV] "
            f"{date.date()} | "
            f"position={self.position:+d} | "
            f"equity=${self.equity:,.2f} | "
            f"reward={self.total_reward:.6f}"
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        pass


# ============================================================
# TEST
# ============================================================

def test_environment():

    print()
    print("=" * 60)
    print("V25 TRADING ENVIRONMENT TEST")
    print("=" * 60)

    env = TradingEnvironment()

    print()
    print(
        f"Rows: {len(env.data)}"
    )

    print(
        f"Features: {len(env.features)}"
    )

    print(
        f"Observation shape: "
        f"{env.observation_space.shape}"
    )

    print(
        f"Actions: "
        f"{env.action_space.n}"
    )

    print(
        f"Training period: "
        f"{env.data['Date'].min().date()} "
        f"→ "
        f"{env.data['Date'].max().date()}"
    )

    env.leakage_check()

    # --------------------------------------------------------
    # RESET TEST
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=42
    )

    print()
    print(
        "Reset successful."
    )

    print(
        f"Observation length: "
        f"{len(observation)}"
    )

    print(
        f"Starting date: "
        f"{info['date'].date()}"
    )

    # --------------------------------------------------------
    # STEP TEST
    # --------------------------------------------------------

    print()
    print(
        "Running 20 simulated steps..."
    )

    for step_number in range(20):

        action = (
            env.action_space.sample()
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"{step_number + 1:02d} | "
            f"action={action} | "
            f"reward={reward:+.6f} | "
            f"equity=${info['equity']:,.2f}"
        )

        if terminated or truncated:

            print(
                "Episode ended."
            )

            break

    env.close()

    print()
    print("=" * 60)
    print("V25 ENVIRONMENT TEST PASSED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_environment()