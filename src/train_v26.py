"""
============================================================
V25 RL LEARNER
============================================================

Research / paper trading ONLY.

Trains a PPO reinforcement-learning agent against the
V25 TradingEnvironment.

Training:
    2015 -> 2023

Evaluation:
    2024+ remains completely held out.

Actions:
    0 = HOLD
    1 = BUY
    2 = SELL
"""

from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from trading_env_v25 import TradingEnvironment


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "v25_ppo"

TOTAL_TIMESTEPS = 500_000

EPISODE_LENGTH = 252

SEED = 42


# ============================================================
# TRAINING CALLBACK
# ============================================================

class ProgressCallback(BaseCallback):

    def __init__(self, print_every=10_000):

        super().__init__()

        self.print_every = print_every
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
                f"[V25] "
                f"Training steps: "
                f"{self.num_timesteps:,}"
            )

        return True


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment():

    env = TradingEnvironment(
        episode_length=EPISODE_LENGTH,
        seed=SEED,
    )

    return env


# ============================================================
# TRAIN
# ============================================================

def train():

    print()
    print("=" * 60)
    print("V25 REINFORCEMENT LEARNING")
    print("=" * 60)

    print()
    print("[V25] Creating environment...")

    env = create_environment()

    print(
        f"[V25] Observations: "
        f"{env.observation_space.shape[0]}"
    )

    print(
        f"[V25] Actions: "
        f"{env.action_space.n}"
    )

    print(
        f"[V25] Training dates: "
        f"{env.data['Date'].min().date()} "
        f"→ "
        f"{env.data['Date'].max().date()}"
    )

    env.leakage_check()

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    print()
    print("[V25] Creating PPO agent...")

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

        seed=SEED,

        device="auto",
    )

    print()
    print(
        "[V25] Starting training..."
    )

    print(
        f"[V25] Timesteps: "
        f"{TOTAL_TIMESTEPS:,}"
    )

    # --------------------------------------------------------
    # LEARNING LOOP
    # --------------------------------------------------------

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ProgressCallback(),
        reset_num_timesteps=True,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    model.save(
        str(MODEL_PATH)
    )

    print()
    print(
        "[V25] Model saved:"
    )

    print(
        f"       {MODEL_PATH}.zip"
    )

    env.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()