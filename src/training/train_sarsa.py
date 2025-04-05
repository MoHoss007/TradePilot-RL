from src.environment.trading_env import TradingEnv
from src.rl_agents.sarsa_agent import SarsaAgent


def train_sarsa_trading_env(
    start_date="2024-01-01",
    end_date="2024-12-30",
    ticker="META",
    window_size=6,
    n_episodes=200,
    n_actions=3,
    include_indicators=False,
    epsilon=0.1,
    alpha=0.1,
    gamma=0.99,
    save_path="trained_sarsa_qtable.pkl",
):
    env = TradingEnv(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        window_size=window_size,
        include_indicators=include_indicators,
    )
    agent = SarsaAgent(n_actions=n_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)

    for ep in range(n_episodes):
        state = env.reset()
        done = False

        action = agent.get_action(state)

        info = {}

        while not done:
            next_state, reward, done, truncated, info = env.step(action)

            agent.inventory = info.get("inventory", [])

            next_action = agent.get_action(next_state) if not done else 0

            state_tuple = tuple(state.astype(int))
            next_state_tuple = (
                tuple(next_state.astype(int)) if next_state.size > 0 else tuple()
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
    train_sarsa_trading_env(include_indicators=False)
