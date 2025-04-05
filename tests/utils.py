import matplotlib.pyplot as plt
from src.environment.trading_env import TradingEnv


def plot_trading_signals(env: TradingEnv):
    """
    Plots the stock price over time and overlays buy and sell signals.

    Parameters:
        env: TradingEnv instance that has attributes:
             - hourlyStockData: iterable of HourlyStockValue with a 'timestamp' attribute.
             - prices: numpy array of closing prices.
             - states_buy: list of indices where buy signals occurred.
             - states_sell: list of indices where sell signals occurred.
    """
    timestamps = [
        env.hourlyStockData[i].timestamp for i in range(len(env.hourlyStockData))
    ]
    prices = env.prices

    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, prices, label="Stock Price", color="blue")

    if env.states_buy:
        buy_timestamps = [timestamps[i] for i in env.states_buy]
        buy_prices = [prices[i] for i in env.states_buy]
        plt.scatter(
            buy_timestamps,
            buy_prices,
            marker="^",
            color="green",
            s=100,
            label="Buy Signal",
        )

    if env.states_sell:
        sell_timestamps = [timestamps[i] for i in env.states_sell]
        sell_prices = [prices[i] for i in env.states_sell]
        plt.scatter(
            sell_timestamps,
            sell_prices,
            marker="v",
            color="red",
            s=100,
            label="Sell Signal",
        )

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Stock Price Over Time with Trading Signals")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show(block=True)
