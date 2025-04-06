import argparse
from typing import List, Tuple, Dict, Any
from src.environment.trading_env import TradingEnv
from src.rl_agents.sarsa_agent import SarsaAgent


def train_sarsa_trading_env(
    start_date: str,
    end_date: str,
    ticker: str,
    window_size: int,
    n_episodes: int,
    n_actions: int,
    include_indicators: bool,
    epsilon: float,
    alpha: float,
    gamma: float,
    save_dir: str,
) -> None:
    env: TradingEnv = TradingEnv(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        window_size=window_size,
        include_indicators=include_indicators,
    )
    agent: SarsaAgent = SarsaAgent(
        n_actions=n_actions, epsilon=epsilon, alpha=alpha, gamma=gamma
    )

    for ep in range(n_episodes):
        state = env.reset()
        done: bool = False

        action: int = agent.get_action(state)

        info: Dict[str, Any] = {}

        while not done:
            next_state, reward, done, truncated, info = env.step(action)

            agent.inventory = info.get("inventory", [])

            next_action: int = agent.get_action(next_state) if not done else 0

            state_tuple: Tuple[int, ...] = tuple(state[0].astype(int))
            next_state_tuple: Tuple[int, ...] = (
                tuple(next_state[0].astype(int)) if next_state.size > 0 else tuple()
            )

            agent.update(
                state_tuple, action, reward, next_state_tuple, next_action, done
            )

            state = next_state
            action = next_action

        print(
            f"Episode {ep+1}/{n_episodes} ended with total profit: {info.get('total_profit', 0)}"
        )
        agent.decay_epsilon()

    save_path = f"{save_dir}/sarsa_{ticker}_{start_date}_to_{end_date}.pkl"
    agent.save_weights(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SARSA agent on trading environment."
    )
    parser.add_argument(
        "--start_date", type=str, default="2023-06-01", help="Start date for training."
    )
    parser.add_argument(
        "--end_date", type=str, default="2024-06-30", help="End date for training."
    )
    parser.add_argument(
        "--ticker", type=str, default="INTC", help="Stock ticker symbol."
    )
    parser.add_argument(
        "--window_size", type=int, default=6, help="Window size for the environment."
    )
    parser.add_argument(
        "--n_episodes", type=int, default=200, help="Number of episodes for training."
    )
    parser.add_argument(
        "--n_actions", type=int, default=3, help="Number of possible actions."
    )
    parser.add_argument(
        "--include_indicators",
        action="store_true",
        help="Include indicators in the environment.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Exploration rate for the agent."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Learning rate for the agent."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for the agent."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="trained_models",
        help="Directory to save trained models.",
    )

    args = parser.parse_args()

    train_sarsa_trading_env(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker=args.ticker,
        window_size=args.window_size,
        n_episodes=args.n_episodes,
        n_actions=args.n_actions,
        include_indicators=args.include_indicators,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        save_dir=args.save_dir,
    )
