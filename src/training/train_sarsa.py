from typing import List, Tuple, Dict, Any
from src.environment.trading_env import TradingEnv
from src.rl_agents.sarsa_agent import SarsaAgent


def train_sarsa_trading_env(
    start_date: str = "2023-06-01",
    end_date: str = "2024-06-30",
    ticker: str = "META",
    window_size: int = 6,
    n_episodes: int = 200,
    n_actions: int = 3,
    include_indicators: bool = False,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.99,
    save_path: str = "trained_models/trained_sarsa_qtable.pkl",
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

    agent.save_weights(save_path)


if __name__ == "__main__":
    train_sarsa_trading_env(include_indicators=True, n_episodes=200)
