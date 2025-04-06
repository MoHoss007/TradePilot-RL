import argparse
from src.rl_agents.sarsa_agent import SarsaAgent
from src.environment.trading_env import TradingEnv
from typing import List, Dict
from tests.utils import plot_trading_signals


def test_sarsa(
    start_date: str,
    end_date: str,
    ticker: str,
    window_size: int,
    n_actions: int,
    qtable_path: str,
    include_indicators: bool,
) -> None:
    env = TradingEnv(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        window_size=window_size,
        include_indicators=include_indicators,
    )
    agent = SarsaAgent(n_actions=n_actions, epsilon=0)
    agent.load_wights(qtable_path)

    state = env.reset()
    done: bool = False

    pending_buys: List[float] = []
    transactions: List[Dict[str, float]] = []

    # Helper function to convert action integer to text.
    def action_text(action: int) -> str:
        return {0: "hold", 1: "buy", 2: "sell"}.get(action, "unknown")

    print("Testing model...")
    info: Dict[str, float] = {}
    while not done:
        current_step: int = env.current_step
        current_price: float = env.prices[current_step]

        action: int = agent.get_action(state)
        act_text: str = action_text(action)

        if action == 1:
            pending_buys.append(current_price)
            print(
                f"Step: {current_step}, Action: {act_text.upper()}, Bought at: {current_price:.2f}"
            )
        elif action == 2:
            if pending_buys:
                bought_price: float = pending_buys.pop(0)
                sell_price: float = current_price
                profit: float = sell_price - bought_price
                transactions.append(
                    {
                        "bought_price": bought_price,
                        "sold_price": sell_price,
                        "profit": profit,
                    }
                )
                print(
                    f"Step: {current_step}, Action: {act_text.upper()}, Sold at: {sell_price:.2f} (Bought at: {bought_price:.2f})"
                )
            else:
                print(
                    f"Step: {current_step}, Action: {act_text.upper()}, No stock to sell."
                )
        else:
            print(f"Step: {current_step}, Action: {act_text.upper()}")

        next_state, reward, done, truncated, info = env.step(action)
        agent.inventory = info.get("inventory", [])

        state = next_state

    if pending_buys:
        for price in pending_buys:
            print(f"Unsold stock bought at: {price:.2f}")

    # Print a summary of all completed transactions.
    print("\n--- TRANSACTION SUMMARY ---")
    for idx, trans in enumerate(transactions, 1):
        print(
            f"Transaction {idx}: Bought at {trans['bought_price']:.2f}, Sold at {trans['sold_price']:.2f}, Profit: {trans['profit']:.2f}"
        )
    print(f"Total Profit: {info.get('total_profit', 0):.2f}")
    plot_trading_signals(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test SARSA agent on trading environment."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2024-07-01",
        help="Start date for the trading environment.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2024-12-30",
        help="End date for the trading environment.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="INTC",
        help="Ticker symbol for the stock.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=6,
        help="Window size for the trading environment.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=3,
        help="Number of actions for the SARSA agent.",
    )
    parser.add_argument(
        "--qtable_path",
        type=str,
        required=True,
        help="Path to the trained Q-table file.",
    )
    parser.add_argument(
        "--include_indicators",
        action="store_true",
        help="Whether to include technical indicators.",
    )

    args = parser.parse_args()

    test_sarsa(
        start_date=args.start_date,
        end_date=args.end_date,
        ticker=args.ticker,
        window_size=args.window_size,
        n_actions=args.n_actions,
        qtable_path=args.qtable_path,
        include_indicators=args.include_indicators,
    )
