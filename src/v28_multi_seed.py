"""
============================================================
V28 MULTI-SEED TRAINING EXPERIMENT
============================================================

RESEARCH / PAPER TRADING ONLY.

Purpose:
    Train fresh PPO agents with multiple independent seeds.

V28 action space:
    0 = HOLD / FLAT
    1 = BUY / LONG

Features:
    85

Training:
    2015-01-01 -> 2023-12-29

Development evaluation:
    2022-01-01 -> 2023-12-29

FINAL HOLDOUT:
    2024+ is NOT touched by this script.

V27 is NOT modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from trading_env_v25 import TradingEnvironment


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("models")

V28_DIR = MODEL_DIR / "v28"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

V28_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SEEDS = [
    101,
    202,
    303,
    404,
    505,
]

TOTAL_TIMESTEPS = 500_000

EPISODE_LENGTH = 252

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-29"

DEV_START = "2022-01-01"
DEV_END = "2023-12-29"


# ============================================================
# V26/V27 COMPATIBLE ENVIRONMENT
# ============================================================

class V28Environment(
    TradingEnvironment
):

    def __init__(
        self,
        *args,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        # ----------------------------------------------------
        # V28 uses the same action space as V26/V27.
        #
        # 0 = HOLD
        # 1 = BUY
        # ----------------------------------------------------

        self.action_space = spaces.Discrete(
            2
        )


# ============================================================
# CALLBACK
# ============================================================

class ProgressCallback(
    BaseCallback
):

    def __init__(
        self,
        seed,
        print_every=10_000
    ):

        super().__init__()

        self.seed = seed

        self.print_every = (
            print_every
        )

        self.last_print = 0

    def _on_step(self):

        if (
            self.num_timesteps
            - self.last_print
            >= self.print_every
        ):

            self.last_print = (
                self.num_timesteps
            )

            print(
                f"[V28][Seed {self.seed}] "
                f"Training steps: "
                f"{self.num_timesteps:,}"
            )

        return True


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment(
    start_date,
    end_date,
    seed
):

    env = V28Environment(

        start_date=start_date,

        end_date=end_date,

        episode_length=EPISODE_LENGTH,

        seed=seed,
    )

    if env.observation_space.shape != (
        85,
    ):

        raise RuntimeError(
            "Expected 85-dimensional "
            "observation space."
        )

    if env.action_space != spaces.Discrete(2):

        raise RuntimeError(
            "Expected Discrete(2) action space."
        )

    return env


# ============================================================
# DEVELOPMENT EVALUATION
# ============================================================

def evaluate_model(
    model,
    seed,
    episodes=5
):

    env = create_environment(
        DEV_START,
        DEV_END,
        seed
    )

    rewards = []

    equities = []

    trade_counts = []

    max_drawdowns = []

    for episode in range(
        episodes
    ):

        observation, info = (
            env.reset(
                seed=seed + episode
            )
        )

        done = False

        total_reward = 0.0

        equity_curve = [
            env.equity
        ]

        while not done:

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            action = int(
                np.asarray(
                    action
                ).item()
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += float(
                reward
            )

            equity_curve.append(
                env.equity
            )

            done = (
                terminated
                or truncated
            )

        rewards.append(
            total_reward
        )

        equities.append(
            env.equity
        )

        trade_counts.append(
            env.trade_count
        )

        curve = np.asarray(
            equity_curve,
            dtype=float
        )

        peak = np.maximum.accumulate(
            curve
        )

        drawdown = (
            curve / peak
        ) - 1.0

        max_drawdowns.append(
            float(drawdown.min())
        )

    env.close()

    return {
        "reward": float(
            np.mean(rewards)
        ),

        "equity": float(
            np.mean(equities)
        ),

        "return": float(
            np.mean(equities)
            / 10_000.0
            - 1.0
        ),

        "trades": float(
            np.mean(trade_counts)
        ),

        "max_dd": float(
            np.mean(max_drawdowns)
        ),
    }


# ============================================================
# TRAIN ONE SEED
# ============================================================

def train_seed(seed):

    print()
    print("=" * 60)

    print(
        f"V28 SEED {seed}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    env = create_environment(
        TRAIN_START,
        TRAIN_END,
        seed
    )

    print()
    print(
        f"[V28][Seed {seed}] "
        f"Observations: "
        f"{env.observation_space.shape}"
    )

    print(
        f"[V28][Seed {seed}] "
        f"Actions: "
        f"{env.action_space}"
    )

    env.leakage_check()

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    print()
    print(
        f"[V28][Seed {seed}] "
        "Creating fresh PPO..."
    )

    model = PPO(

        policy="MlpPolicy",

        env=env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,

        seed=seed,

        device="cpu",
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print(
        f"[V28][Seed {seed}] "
        f"Training "
        f"{TOTAL_TIMESTEPS:,} steps..."
    )

    model.learn(

        total_timesteps=TOTAL_TIMESTEPS,

        callback=ProgressCallback(
            seed
        ),

        reset_num_timesteps=True,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    model_path = (
        V28_DIR
        / f"v28_seed_{seed}"
    )

    model.save(
        str(model_path)
    )

    print()
    print(
        f"[V28][Seed {seed}] "
        "Model saved:"
    )

    print(
        f"    {model_path}.zip"
    )

    # --------------------------------------------------------
    # DEVELOPMENT EVALUATION
    # --------------------------------------------------------

    print()
    print(
        f"[V28][Seed {seed}] "
        "Running development evaluation..."
    )

    results = evaluate_model(
        model,
        seed
    )

    print()
    print(
        f"[V28][Seed {seed}] "
        "DEVELOPMENT RESULTS"
    )

    print(
        f"    Reward: "
        f"{results['reward']:+.6f}"
    )

    print(
        f"    Equity: "
        f"${results['equity']:,.2f}"
    )

    print(
        f"    Return: "
        f"{results['return'] * 100:+.2f}%"
    )

    print(
        f"    Avg max DD: "
        f"{results['max_dd'] * 100:.2f}%"
    )

    print(
        f"    Trades: "
        f"{results['trades']:.1f}"
    )

    env.close()

    return {
        "seed": seed,
        "model": str(model_path) + ".zip",
        **results,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V28 MULTI-SEED PPO EXPERIMENT")
    print("=" * 60)

    print()
    print(
        "Fresh models:"
    )

    print(
        "    " +
        ", ".join(
            str(seed)
            for seed in SEEDS
        )
    )

    print()
    print(
        f"Timesteps per seed: "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Total planned timesteps: "
        f"{TOTAL_TIMESTEPS * len(SEEDS):,}"
    )

    print()
    print(
        "FINAL HOLDOUT 2024+ = LOCKED"
    )

    print(
        "V27 = LOCKED"
    )

    # --------------------------------------------------------
    # RUN ALL SEEDS
    # --------------------------------------------------------

    results = []

    for seed in SEEDS:

        result = train_seed(
            seed
        )

        results.append(
            result
        )

        # Save progress after every seed.
        pd.DataFrame(
            results
        ).to_csv(
            V28_DIR
            / "v28_seed_results.csv",
            index=False
        )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "reward",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 60)
    print("V28 MULTI-SEED RESULTS")
    print("=" * 60)

    print()

    display_df = results_df.copy()

    display_df["return"] = (
        display_df["return"]
        .map(
            lambda x:
            f"{x * 100:+.2f}%"
        )
    )

    display_df["max_dd"] = (
        display_df["max_dd"]
        .map(
            lambda x:
            f"{x * 100:.2f}%"
        )
    )

    display_df["reward"] = (
        display_df["reward"]
        .map(
            lambda x:
            f"{x:+.6f}"
        )
    )

    display_df["equity"] = (
        display_df["equity"]
        .map(
            lambda x:
            f"${x:,.2f}"
        )
    )

    display_df["trades"] = (
        display_df["trades"]
        .map(
            lambda x:
            f"{x:.1f}"
        )
    )

    print(
        display_df[
            [
                "seed",
                "reward",
                "equity",
                "return",
                "max_dd",
                "trades",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST SEED
    # --------------------------------------------------------

    best = results_df.iloc[0]

    print()
    print("=" * 60)
    print("BEST V28 DEVELOPMENT MODEL")
    print("=" * 60)

    print()

    print(
        f"Seed: "
        f"{int(best['seed'])}"
    )

    print(
        f"Reward: "
        f"{best['reward']:+.6f}"
    )

    print(
        f"Equity: "
        f"${best['equity']:,.2f}"
    )

    print(
        f"Return: "
        f"{best['return'] * 100:+.2f}%"
    )

    print(
        f"Average max DD: "
        f"{best['max_dd'] * 100:.2f}%"
    )

    print(
        f"Trades: "
        f"{best['trades']:.1f}"
    )

    print()

    print(
        f"Model:"
    )

    print(
        f"    {best['model']}"
    )

    # --------------------------------------------------------
    # CONSISTENCY
    # --------------------------------------------------------

    rewards = results_df[
        "reward"
    ].to_numpy()

    returns = results_df[
        "return"
    ].to_numpy()

    print()
    print("=" * 60)
    print("SEED CONSISTENCY")
    print("=" * 60)

    print()

    print(
        f"Mean reward: "
        f"{np.mean(rewards):+.6f}"
    )

    print(
        f"Reward std: "
        f"{np.std(rewards):.6f}"
    )

    print(
        f"Mean return: "
        f"{np.mean(returns) * 100:+.2f}%"
    )

    print(
        f"Return std: "
        f"{np.std(returns) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE FINAL RESULTS
    # --------------------------------------------------------

    output = (
        V28_DIR
        / "v28_seed_results.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print()
    print(
        f"Results saved:"
    )

    print(
        f"    {output}"
    )

    print()
    print("=" * 60)
    print("V28 MULTI-SEED EXPERIMENT COMPLETE")
    print("=" * 60)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "    Do NOT benchmark V28 on 2024+ "
        "until we select the candidate."
    )

    print(
        "    Do NOT modify v27_best_FROZEN.zip."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()
