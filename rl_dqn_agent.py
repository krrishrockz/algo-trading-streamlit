# rl_dqn_agent.py
# ------------------------------------------------------------
# DQN trading with Gymnasium-compatible environment (3 actions).
# Normalized observations, optional transaction cost, epsilon-greedy eval.
# Works with Stable-Baselines3 v2.x (gymnasium API).
# ------------------------------------------------------------
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

# Ensure logs directory exists (works locally and on Streamlit Cloud)
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "rl_dqn_agent.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StockTradingEnv(gym.Env):
    """
    - Actions: 0=Hold, 1=Buy, 2=Sell
    - Observation: normalized [Open, High, Low, Close, Volume] (float32)
    - Reward: per-step change in net worth (Δ net worth)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 10_000.0,
        normalize_obs: bool = True,
        fee_rate: float = 0.0005,  # 5 bps per trade
    ):
        super().__init__()

        # --- REQUIRED COLUMNS + COERCION ---
        req = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Env: missing OHLCV columns: {missing}")

        keep = req + (["Date"] if "Date" in df.columns else [])
        self.df = df[keep].copy().reset_index(drop=True)

        # Force numeric, replace inf with NaN, forward/back fill
        for c in req:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        self.df[req] = self.df[req].replace([np.inf, -np.inf], np.nan)
        self.df[req] = self.df[req].ffill().bfill()
        if self.df[req].isna().any().any():
            raise ValueError("Env: NaNs remain in OHLCV after fill; cannot proceed.")

        # If Volume is entirely zero, add epsilon
        if (self.df["Volume"] == 0).all():
            self.df["Volume"] = 1e-6

        self.n_steps = len(self.df)
        if self.n_steps < 2:
            raise ValueError("DataFrame must have at least 2 rows.")

        self.initial_cash = float(initial_cash)
        self.normalize_obs = bool(normalize_obs)
        self.fee_rate = float(fee_rate)

        # Precompute normalization scales
        self._price_scale = float(max(self.df["Close"].iloc[0], 1.0))
        self._vol_scale = float(max(self.df["Volume"].max(), 1.0))

        # Episode state
        self.current_step = 0
        self.balance = self.initial_cash
        self.shares_held = 0
        self.net_worth = self.initial_cash
        self.prev_net_worth = self.initial_cash
        self.max_steps = self.n_steps - 1
        self.trades = []

        # Spaces
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(None)

    def _obs_raw(self):
        r = self.df.iloc[self.current_step]
        return np.array([r["Open"], r["High"], r["Low"], r["Close"], r["Volume"]], dtype=np.float32)

    def _normalize(self, obs):
        if not self.normalize_obs:
            return obs
        # price normalization by first close; volume by max volume
        p = obs[:4] / self._price_scale
        v = np.array([obs[4] / self._vol_scale], dtype=np.float32)
        return np.concatenate([p.astype(np.float32), v]).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._normalize(self._obs_raw())

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.current_step = 0
        self.balance = self.initial_cash
        self.shares_held = 0
        self.net_worth = self.initial_cash
        self.prev_net_worth = self.initial_cash
        self.trades = []

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        # 0 = Hold, 1 = Buy (1 share), 2 = Sell (1 share)
        price_now = float(self.df.iloc[self.current_step]["Close"])
        trade_executed = None

        if action == 1 and self.balance >= price_now:  # Buy
            fee = price_now * self.fee_rate
            total_cost = price_now + fee
            if self.balance >= total_cost:
                self.shares_held += 1
                self.balance -= total_cost
                trade_executed = ("Buy", self.current_step, price_now)

        elif action == 2 and self.shares_held > 0:  # Sell
            fee = price_now * self.fee_rate
            self.shares_held -= 1
            self.balance += (price_now - fee)
            trade_executed = ("Sell", self.current_step, price_now)

        # Advance time
        self.current_step += 1

        # Mark-to-market at current (clipped) index
        price_index = min(self.current_step, self.max_steps)
        mark_price = float(self.df.iloc[price_index]["Close"])

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * mark_price

        if trade_executed:
            self.trades.append(
                {
                    "step": trade_executed[1],
                    "type": trade_executed[0],
                    "price": trade_executed[2],
                    "net_worth": self.net_worth,
                }
            )

        # Reward: per-step Δ net worth
        reward = float(self.net_worth - self.prev_net_worth)

        # Episode termination at end of data
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs() if not terminated else self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step} | "
            f"Net Worth: {self.net_worth:.2f} | "
            f"Shares: {self.shares_held} | "
            f"Balance: {self.balance:.2f}"
        )


# -----------------------------
# Train DQN Agent on Stock Data
# -----------------------------
def train_dqn_agent(df: pd.DataFrame, total_timesteps: int = 20_000, seed: int | None = 42):
    """
    Create the env, train a DQN model, and return it.
    df must have columns: ['Open','High','Low','Close','Volume'] (and optionally 'Date').
    """
    env = StockTradingEnv(df, normalize_obs=True, fee_rate=0.0005)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        # Slightly higher exploration during learning so policy isn't "always hold"
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.3,   # ↑ exploration during training
        exploration_final_eps=0.10, # keep some exploration
    )
    model.learn(total_timesteps=total_timesteps)
    return model


# -------------------------------------
# Simulate Trading with Trained DQN Agent
# -------------------------------------
def simulate_trading(df: pd.DataFrame, model: DQN, eval_epsilon: float = 0.10):
    """
    Roll out a trained model through the environment with epsilon-greedy evaluation
    (so you actually see trades if the policy is timid).
    Returns:
      rewards (list of per-step rewards),
      worths  (list of net worth over time),
      trade_log (DataFrame with executed trades)
    """
    env = StockTradingEnv(df, normalize_obs=True, fee_rate=0.0005)
    obs, info = env.reset()
    rewards = []
    worths = [env.net_worth]  # initial net worth
    trade_log = pd.DataFrame(columns=["step", "type", "price", "net_worth"])

    rng = np.random.RandomState(123)

    for _ in range(len(df)):
        # epsilon-greedy at eval time
        if rng.rand() < max(0.0, min(1.0, eval_epsilon)):
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(float(reward))
        worths.append(env.net_worth)

        if env.trades:
            trade_log = pd.concat([trade_log, pd.DataFrame(env.trades)], ignore_index=True)
            env.trades = []  # clear after logging

        if terminated or truncated:
            break

    # Pad worths if short (safety)
    if len(worths) < len(df):
        worths.extend([worths[-1]] * (len(df) - len(worths)))

    return rewards, worths, trade_log


# ============================
# ✅ DQN Evaluation Utilities
# ============================
def evaluate_dqn_performance(trade_log: pd.DataFrame):
    if trade_log.empty:
        return {
            "total_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "final_net_worth": 10_000.0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
        }

    buy_prices = []
    profits = []
    for _, trade in trade_log.iterrows():
        if trade["type"] == "Buy":
            buy_prices.append(float(trade["price"]))
        elif trade["type"] == "Sell" and buy_prices:
            buy_price = buy_prices.pop(0)
            sell_price = float(trade["price"])
            profits.append(sell_price - buy_price)

    win_rate = float(np.mean([p > 0 for p in profits])) if profits else 0.0
    avg_profit = float(np.mean(profits)) if profits else 0.0

    return {
        "total_trades": int(len(trade_log)),
        "buy_trades": int((trade_log["type"] == "Buy").sum()),
        "sell_trades": int((trade_log["type"] == "Sell").sum()),
        "final_net_worth": float(trade_log["net_worth"].iloc[-1]) if not trade_log.empty else 10_000.0,
        "win_rate": round(win_rate * 100, 2),
        "avg_profit": round(avg_profit, 2),
    }


# ------------------------------
# Plot utilities (matplotlib / plotly)
# ------------------------------
def plot_dqn_reward_curve(rewards, title="DQN Reward Curve"):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Reward")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Per-step Reward (Δ Net Worth)")
    plt.grid(True)
    plt.legend()
    return plt.gcf()


def plot_dqn_results(df: pd.DataFrame, worths, trade_log: pd.DataFrame, title="DQN Strategy"):
    """
    Plot: Net Worth (left y) + Close Price (right y) with buy/sell markers on the price axis.
    This fixes the 'price looks like 0' issue when net worth is ~₹10–20k and price is ~₹100–1000.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_date = df["Date"] if "Date" in df.columns else pd.RangeIndex(start=0, stop=len(df), step=1)
    price = df["Close"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Net worth (left axis)
    fig.add_trace(
        go.Scatter(x=x_date, y=worths[: len(x_date)], name="Net Worth", line=dict(color="green", width=2)),
        secondary_y=False,
    )
    # Close price (right axis)
    fig.add_trace(
        go.Scatter(x=x_date, y=price, name="Close Price", line=dict(color="royalblue", width=1)),
        secondary_y=True,
    )

    # Buy/Sell markers (on price axis so they’re not squashed at 0)
    if not trade_log.empty:
        buys = trade_log[trade_log["type"] == "Buy"]
        sells = trade_log[trade_log["type"] == "Sell"]
        if "Date" in df.columns and "step" in buys.columns:
            buy_x = buys["step"].apply(lambda i: df["Date"].iloc[int(i)])
            sell_x = sells["step"].apply(lambda i: df["Date"].iloc[int(i)])
        else:
            buy_x = buys["step"]
            sell_x = sells["step"]

        fig.add_trace(
            go.Scatter(
                x=buy_x, y=buys["price"], mode="markers",
                marker=dict(symbol="triangle-up", size=9, color="green"),
                name="Buy"
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=sell_x, y=sells["price"], mode="markers",
                marker=dict(symbol="triangle-down", size=9, color="red"),
                name="Sell"
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Value (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Close Price", secondary_y=True)

    # Add a small padding on the price y-axis
    y_min = float(price.min())
    y_max = float(price.max())
    pad = max((y_max - y_min) * 0.05, 0.5)
    fig.update_yaxes(range=[y_min - pad, y_max + pad], secondary_y=True)

    return fig



def plot_dqn_net_worth(worths, title="DQN Net Worth"):
    plt.figure(figsize=(10, 4))
    plt.plot(worths, label="Net Worth", color="green")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("₹")
    plt.grid(True)
    plt.legend()
    return plt.gcf()


# Required libraries:
# stable-baselines3>=2.0.0
# gymnasium>=0.29.0
# shimmy>=1.3.0
# torch>=2.2.0
# numpy, pandas, matplotlib, streamlit, plotly


