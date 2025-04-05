import numpy as np
import pickle
from typing import Tuple
from src.rl_agents.sarsa_agent import SarsaAgent
from src.environment.trading_env import TradingEnv
from src.training.train_sarsa import train_sarsa_trading_env


def test():
    # Set parameters consistent with training.
    start_date = "2025-01-01"
    end_date = "2025-04-30"
    ticker = "META"
    window_size = 6  # Must match the window_size used during training.
    n_actions = 3  # Actions: 0 = hold, 1 = buy, 2 = sell.

    # Initialize the environment and the agent.
    env = TradingEnv(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        window_size=window_size,
        include_indicators=True,
    )
    agent = SarsaAgent(n_actions=n_actions)
    # Load the previously trained Q-table.
    agent.load_wights("trained_sarsa_qtable.pkl")

    # Reset the environment to start a new episode.
    state = env.reset()
    done = False

    print("Testing model...")
    while not done:
        # Agent selects an action (with its own logic preventing an invalid sell).
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        agent.inventory = info.get("inventory", [])

        # Print the current step, chosen action, and cumulative profit.
        print(
            f"Step: {info.get('current_step')}, Action: {action}, Profit: {info.get('total_profit')}"
        )

        # Move to the next state.
        state = next_state


if __name__ == "__main__":
    test()
