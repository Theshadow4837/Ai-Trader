"""
============================================================
V26 TRADING ENVIRONMENT
============================================================

Research / paper-trading environment ONLY.

V26 changes from V25:
    - 2 actions instead of 3
    - 0 = FLAT
    - 1 = LONG
    - No short selling
    - Reward uses the next market day's return
    - Future returns are NEVER observations
    - Training period remains 2015-2023

Timing:

    observation(t)
          |
          v
      action(t)
          |
          v
   market return t -> t+1
          |
          v
       reward
          |
          v
   observation(t+1)
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path(
    "data/market_features_v14.csv"
)

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-29"

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005

DEFAULT_EPISODE_LENGTH = 252


# ============================================================
# ACTIONS
# ============================================================

FLAT = 0
LONG = 1


# ============================================================
# FORBIDDEN OBSERVATION WORDS
# ============================================================

FORBIDDEN_FEATURE_WORDS = {
    "future",
    "target",
    "label",
    "reward",
}


# ============================================================
# ENVIRONMENT
# ============================================================

class TradingEnvironmentV26(gym.Env):

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

        self.data_file = Path(
            data_file
        )

        self.start_date = pd.Timestamp(
            start_date
        )

        self.end_date = pd.Timestamp(
            end_date
        )

        self.episode_length = int(
            episode_length
        )

        self.transaction_cost = float(
            transaction_cost
        )

        self.initial_capital = float(
            initial_capital
        )

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
        # REWARD SOURCE
        #
        # Prefer future_1d_return if available.
        #
        # Otherwise derive the next-day SPY return.
        #
        # This value is INTERNAL ONLY.
        # ----------------------------------------------------

        if "future_1d_return" in self.data.columns:

            self.reward_returns = (
                self.data[
                    "future_1d_return"
                ]
                .astype(np.float32)
                .to_numpy()
                .copy()
            )

        elif "SPY_return_1d" in self.data.columns:

            self.reward_returns = (
                self.data[
                    "SPY_return_1d"
                ]
                .shift(-1)
                .astype(np.float32)
                .to_numpy()
                .copy()
            )

            self.reward_returns[-1] = 0.0

        else:

            raise ValueError(
                "Dataset contains neither "
                "future_1d_return nor "
                "SPY_return_1d."
            )

        # ----------------------------------------------------
        # SELECT PERIOD
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

        # Reward array must use exactly the same rows.
        #
        # Rebuild it after date filtering so the reward and
        # observation indices always remain aligned.
        # ----------------------------------------------------

        if "future_1d_return" in self.data.columns:

            self.reward_returns = (
                self.data[
                    "future_1d_return"
                ]
                .astype(np.float32)
                .to_numpy()
                .copy()
            )

        else:

            self.reward_returns = (
                self.data[
                    "SPY_return_1d"
                ]
                .shift(-1)
                .astype(np.float32)
                .to_numpy()
                .copy()
            )

            self.reward_returns[-1] = 0.0

        if len(self.data) < 100:

            raise ValueError(
                "Not enough training data."
            )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        self.features = (
            self._find_features()
        )

        if not self.features:

            raise ValueError(
                "No usable features were found."
            )

        # ----------------------------------------------------
        # REMOVE MISSING OBSERVATIONS
        # ----------------------------------------------------

        valid_mask = (
            ~self.data[
                self.features
            ]
            .isna()
            .any(axis=1)
        )

        self.data = (
            self.data.loc[
                valid_mask
            ]
            .reset_index(drop=True)
        )

        self.reward_returns = (
            self.reward_returns[
                valid_mask.to_numpy()
            ]
        )

        # ----------------------------------------------------
        # OBSERVATION MATRIX
        # ----------------------------------------------------

        self.X = (
            self.data[
                self.features
            ]
            .astype(np.float32)
            .to_numpy()
        )

        # ----------------------------------------------------
        # NORMALIZATION
        #
        # Statistics come ONLY from this environment's
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
            (self.X - self.mean)
            / self.std
        )

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

        self.action_space = spaces.Discrete(
            2
        )

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(
                len(self.features),
            ),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.current_step = 0
        self.episode_start = 0
        self.episode_end = 0

        self.position = FLAT

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

            features.append(
                column
            )

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

                    leaks.append(
                        feature
                    )

                    break

        if leaks:

            raise RuntimeError(
                "LEAKAGE DETECTED:\n"
                + "\n".join(leaks)
            )

        print(
            "[V26] Leakage check: PASSED"
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

        super().reset(
            seed=seed
        )

        maximum_start = max(
            0,
            len(self.X)
            - self.episode_length
            - 2
        )

        if maximum_start > 0:

            self.episode_start = int(
                self.np_random.integers(
                    0,
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

        self.position = FLAT

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

    def step(
        self,
        action
    ):

        action = int(
            action
        )

        if not self.action_space.contains(
            action
        ):

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
        # ACTION
        # ----------------------------------------------------

        if action == LONG:

            new_position = LONG

        else:

            new_position = FLAT

        # ----------------------------------------------------
        # POSITION CHANGE
        # ----------------------------------------------------

        position_changed = (
            new_position
            != self.position
        )

        cost = 0.0

        if position_changed:

            cost = (
                self.transaction_cost
            )

            self.trade_count += 1

        # ----------------------------------------------------
        # NEXT-DAY RETURN
        #
        # This is the reward source for action(t).
        # It is NEVER part of observation(t).
        # ----------------------------------------------------

        market_return = float(
            self.reward_returns[
                self.current_step
            ]
        )

        # ----------------------------------------------------
        # STRATEGY RETURN
        # ----------------------------------------------------

        if new_position == LONG:

            strategy_return = (
                market_return
            )

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
            1.0
            + strategy_return
        )

        self.position = (
            new_position
        )

        self.capital = (
            self.equity
        )

        # ----------------------------------------------------
        # LOG REWARD
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

        self.total_reward += (
            reward
        )

        # ----------------------------------------------------
        # ADVANCE
        # ----------------------------------------------------

        next_step = (
            self.current_step + 1
        )

        terminated = (
            next_step
            >= self.episode_end
        )

        if next_step >= len(self.X):

            terminated = True

            next_step = (
                len(self.X) - 1
            )

        self.current_step = (
            next_step
        )

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
            f"[V26] "
            f"{date.date()} | "
            f"position={self.position} | "
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
    print("V26 TRADING ENVIRONMENT TEST")
    print("=" * 60)

    env = TradingEnvironmentV26()

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

    observation, info = (
        env.reset(seed=42)
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

    print()
    print(
        "Running 20 simulated steps..."
    )

    for step in range(20):

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
            f"{step + 1:02d} | "
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
    print("V26 ENVIRONMENT TEST PASSED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_environment()