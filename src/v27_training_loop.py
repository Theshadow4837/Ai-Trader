"""
============================================================
V27 SELF-IMPROVEMENT TRAINING LOOP
============================================================

RESEARCH / PAPER TRADING ONLY.

V26 compatibility:
    Action 0 = HOLD
    Action 1 = BUY

V26 observation space:
    85 features

The final 2024+ holdout is NOT used for model selection.

Training loop:

    V26
      ↓
    train
      ↓
    development evaluation
      ↓
    keep best
      ↓
    train again
      ↓
    repeat
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from trading_env_v25 import TradingEnvironment


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("models")
CHECKPOINT_DIR = MODEL_DIR / "v27_checkpoints"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

START_MODEL = MODEL_DIR / "v26_ppo.zip"

BEST_MODEL = MODEL_DIR / "v27_best.zip"

NUM_ROUNDS = 10

STEPS_PER_ROUND = 100_000

EPISODE_LENGTH = 252

SEED = 42

DEV_START = "2022-01-01"

DEV_END = "2023-12-29"


# ============================================================
# 2-ACTION ENVIRONMENT
# ============================================================

class V26CompatibleEnvironment(
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
        # V26 ONLY HAS TWO ACTIONS:
        #
        # 0 = HOLD
        # 1 = BUY
        #
        # Do NOT expose SELL to the V26 policy.
        # ----------------------------------------------------

        from gymnasium import spaces

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
        print_every=10_000
    ):

        super().__init__()

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
                f"[V27] Training steps: "
                f"{self.num_timesteps:,}"
            )

        return True


# ============================================================
# TRAINING ENVIRONMENT
# ============================================================

def create_training_env():

    return V26CompatibleEnvironment(

        start_date="2015-01-01",

        end_date="2023-12-29",

        episode_length=EPISODE_LENGTH,

        seed=SEED,
    )


# ============================================================
# DEVELOPMENT ENVIRONMENT
# ============================================================

def create_dev_env():

    return V26CompatibleEnvironment(

        start_date=DEV_START,

        end_date=DEV_END,

        episode_length=EPISODE_LENGTH,

        seed=SEED,
    )


# ============================================================
# DEVELOPMENT EVALUATION
# ============================================================

def evaluate_development(
    model,
    episodes=5
):

    env = create_dev_env()

    rewards = []

    final_equities = []

    trade_counts = []

    for episode in range(
        episodes
    ):

        observation, info = env.reset(
            seed=SEED + episode
        )

        done = False

        total_reward = 0.0

        while not done:

            action, _ = model.predict(

                observation,

                deterministic=True
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

            done = (
                terminated
                or truncated
            )

        rewards.append(
            total_reward
        )

        final_equities.append(
            float(
                env.equity
            )
        )

        trade_counts.append(
            int(
                env.trade_count
            )
        )

    env.close()

    return (
        float(
            np.mean(rewards)
        ),
        float(
            np.mean(final_equities)
        ),
        float(
            np.mean(trade_counts)
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V27 SELF-IMPROVEMENT LOOP")
    print("=" * 60)

    # --------------------------------------------------------
    # VERIFY MODEL
    # --------------------------------------------------------

    if not START_MODEL.exists():

        raise FileNotFoundError(
            f"Missing model: "
            f"{START_MODEL}"
        )

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    print()
    print(
        "[V27] Creating compatible environment..."
    )

    env = create_training_env()

    print(
        f"[V27] Environment observations: "
        f"{env.observation_space}"
    )

    print(
        f"[V27] Environment actions: "
        f"{env.action_space}"
    )

    # --------------------------------------------------------
    # LOAD V26
    # --------------------------------------------------------

    print()
    print(
        "[V27] Loading V26..."
    )

    model = PPO.load(

        str(START_MODEL),

        env=env,

        device="cpu",
    )

    # --------------------------------------------------------
    # VERIFY COMPATIBILITY
    # --------------------------------------------------------

    print()
    print(
        "[V27] Model action space:"
    )

    print(
        f"       {model.action_space}"
    )

    print(
        "[V27] Environment action space:"
    )

    print(
        f"       {env.action_space}"
    )

    if (
        model.action_space
        != env.action_space
    ):

        raise RuntimeError(
            "V26/V27 action-space mismatch."
        )

    print()
    print(
        "[V27] Action-space compatibility: PASSED"
    )

    env.leakage_check()

    # --------------------------------------------------------
    # INITIAL SCORE
    # --------------------------------------------------------

    print()
    print(
        "[V27] Evaluating starting V26..."
    )

    (
        best_reward,
        best_equity,
        best_trades,
    ) = evaluate_development(
        model
    )

    print()
    print(
        "[V27] Starting development score:"
    )

    print(
        f"       Reward: "
        f"{best_reward:+.6f}"
    )

    print(
        f"       Equity: "
        f"${best_equity:,.2f}"
    )

    print(
        f"       Trades: "
        f"{best_trades:.1f}"
    )

    # --------------------------------------------------------
    # SAVE INITIAL BEST
    # --------------------------------------------------------

    model.save(
        str(BEST_MODEL)
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for round_number in range(
        1,
        NUM_ROUNDS + 1
    ):

        print()
        print("=" * 60)

        print(
            f"V27 ROUND "
            f"{round_number}/{NUM_ROUNDS}"
        )

        print("=" * 60)

        print()
        print(
            f"[V27] Training "
            f"{STEPS_PER_ROUND:,} steps..."
        )

        model.learn(

            total_timesteps=STEPS_PER_ROUND,

            callback=ProgressCallback(),

            reset_num_timesteps=False,
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        checkpoint = (
            CHECKPOINT_DIR
            / f"v27_round_{round_number}"
        )

        model.save(
            str(checkpoint)
        )

        print()
        print(
            f"[V27] Saved checkpoint:"
        )

        print(
            f"       {checkpoint}.zip"
        )

        # ----------------------------------------------------
        # DEVELOPMENT EVALUATION
        # ----------------------------------------------------

        (
            reward,
            equity,
            trades,
        ) = evaluate_development(
            model
        )

        print()
        print(
            "[V27] Development result:"
        )

        print(
            f"       Reward: "
            f"{reward:+.6f}"
        )

        print(
            f"       Equity: "
            f"${equity:,.2f}"
        )

        print(
            f"       Trades: "
            f"{trades:.1f}"
        )

        # ----------------------------------------------------
        # KEEP BEST
        # ----------------------------------------------------

        if reward > best_reward:

            print()
            print(
                "🔥 [V27] NEW BEST MODEL"
            )

            print(
                f"    {reward:+.6f}"
                f" > "
                f"{best_reward:+.6f}"
            )

            best_reward = reward

            best_equity = equity

            best_trades = trades

            model.save(
                str(BEST_MODEL)
            )

            print()
            print(
                f"[V27] Best saved:"
            )

            print(
                f"       {BEST_MODEL}"
            )

        else:

            print()
            print(
                "[V27] No improvement."
            )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    env.close()

    print()
    print("=" * 60)
    print("V27 TRAINING LOOP COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Best development reward: "
        f"{best_reward:+.6f}"
    )

    print(
        f"Best development equity: "
        f"${best_equity:,.2f}"
    )

    print()
    print(
        f"Best model:"
    )

    print(
        f"    {BEST_MODEL}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "    2024+ was NOT used "
        "for model selection."
    )

    print()
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()