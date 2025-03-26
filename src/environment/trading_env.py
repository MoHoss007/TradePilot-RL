import gym
from gym import spaces
from src.data_loader.stock_data import HourlyStockData
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        start_date: str,
        end_date: str,
        ticker: str,
        window_size: int = 6,  # number of prices to consider (producing window_size-1 differences)
        tc: float = 0.05 / 100,
        render_mode: Optional[str] = None,
        apply_sigmoid: bool = False,
        include_indicators: bool = False,
    ) -> None:
        """
        Initialize the Trading Environment.

        :param start_date: Start date for historical data.
        :param end_date: End date for historical data.
        :param ticker: Stock ticker symbol.
        :param window_size: Number of prices to use (producing window_size-1 differences).
        :param tc: Transaction cost fraction.
        :param render_mode: Rendering mode (e.g., "human").
        :param apply_sigmoid: Flag to determine if sigmoid should be applied to differences.
        :param include_indicators: Flag to include technical indicators in the state.
        """
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.window_size = window_size
        self.tc = tc
        self.render_mode = render_mode
        self.apply_sigmoid = apply_sigmoid
        self.include_indicators = include_indicators

        self.hourlyStockData = HourlyStockData(
            self.ticker, self.start_date, self.end_date
        )
        self.prices = np.array([x.close for x in self.hourlyStockData])

        self.action_space = spaces.Discrete(3)

        # (window_size-1) differences, plus 4 indicators if include_indicators is True.
        extra_dim = 4 if self.include_indicators else 0
        obs_dim = (self.window_size - 1) + extra_dim

        if self.apply_sigmoid:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(1, obs_dim), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, obs_dim), dtype=np.float32
            )

        # Trading-related variables.
        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        self.current_step = self.window_size - 1
        self._episode_ended = False

    def _sigmoid(self, x: float) -> float:
        """Apply the sigmoid function to x."""
        return 1 / (1 + np.exp(-x))

    def _get_state(self, t: int) -> np.ndarray:
        """
        Build a state representation using a sliding window of closing prices.
        Pads the window with the first price if needed and computes the differences
        between consecutive prices. Optionally applies a sigmoid transformation,
        and if include_indicators is True, appends the current technical indicators.
        """
        n = self.window_size
        d = t - n + 1
        if d >= 0:
            block = self.prices[d : t + 1]
        else:
            block = np.concatenate(
                (np.full((-d,), self.prices[0]), self.prices[0 : t + 1])
            )
        res = []
        for i in range(n - 1):
            diff = block[i + 1] - block[i]
            if self.apply_sigmoid:
                res.append(self._sigmoid(diff))
            else:
                res.append(diff)
        state = np.array(res, dtype=np.float32)
        if self.include_indicators:
            current_data = self.hourlyStockData[t]
            # Append indicators (replacing None with 0.0).
            indicators = np.array(
                [
                    current_data.macd if current_data.macd is not None else 0.0,
                    current_data.ema if current_data.ema is not None else 0.0,
                    current_data.sma if current_data.sma is not None else 0.0,
                    current_data.rsi if current_data.rsi is not None else 0.0,
                ],
                dtype=np.float32,
            )
            state = np.concatenate((state, indicators))
        return state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        super().reset(seed=seed)
        self.hourlyStockData = HourlyStockData(
            self.ticker, self.start_date, self.end_date
        )
        self.prices = np.array([x.close for x in self.hourlyStockData])
        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        self.current_step = self.window_size - 1
        self._episode_ended = False
        obs = self._get_state(self.current_step)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._episode_ended:
            raise RuntimeError("Episode is done. Call reset() before step().")

        reward = 0.0
        if action == 1:  # Buy action.
            self.inventory.append(self.prices[self.current_step])
            self.states_buy.append(self.current_step)
        elif action == 2:  # Sell action.
            if len(self.inventory) > 0:
                bought_price = self.inventory.pop(0)
                profit = self.prices[self.current_step] - bought_price
                reward = profit if profit > 0 else 0.0
                self.total_profit += profit
                self.states_sell.append(self.current_step)
            else:
                reward = 0.0

        self.current_step += 1
        done = False
        if self.current_step >= len(self.prices):
            done = True
            self._episode_ended = True

        obs = self._get_state(self.current_step) if not done else np.array([[]])
        current_data = (
            self.hourlyStockData[self.current_step]
            if self.current_step < len(self.hourlyStockData)
            else None
        )
        info = {
            "current_step": self.current_step,
            "date": current_data.timestamp if current_data else None,
            "total_profit": self.total_profit,
            "inventory": self.inventory,
        }
        return obs, reward, done, False, info

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            print(
                f"Step: {self.current_step}, Inventory: {self.inventory}, Total Profit: {self.total_profit}"
            )
