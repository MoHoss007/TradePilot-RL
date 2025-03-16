import gym
from gym import spaces
from data_loader.stock_data import HourlyStockData
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        start_date: str,
        end_date: str,
        ticker: str,
        tc: float = 0.05 / 100,
        n_lags: int = 5,
        n_future: int = 0,
        render_mode: Optional[str] = None,
        include_indicators: bool = True,
    ) -> None:
        """
        Initialize the TradingEnv.

        :param start_date: Start date for historical data
        :param end_date: End date for historical data
        :param ticker: Stock ticker symbol
        :param tc: Transaction cost as a fraction (e.g., 0.0005 = 0.05%)
        :param n_lags: Number of past returns to include in each observation
        :param n_future: Number of future returns to leak into the observation
                         (0 for no future leakage -- best practice)
        :param render_mode: Optional, can be "human" or None
        :param include_indicators: If True, observation will include the current step's
                                   MACD, EMA, RSI, etc.
        """
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.tc = tc
        self.n_lags = n_lags
        self.n_future = n_future
        self.render_mode = render_mode
        self.include_indicators = include_indicators

        self.hourlyStockData = HourlyStockData(
            self.ticker, self.start_date, self.end_date
        )

        # Define action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        obs_size = self.n_lags + self.n_future
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Current position (0=hold, 1=buy, 2=sell)
        self.action = 0
        self.current_step = self.n_lags

        self._episode_ended = False

    def _get_lagged_returns(self, index: int) -> np.ndarray:
        """
        Return an array of shape (n_lags,) with
          ret(t-1), ret(t-2), ..., ret(t-n_lags).
        """
        lagged = []
        for i in range(self.n_lags):
            curr_price = self.hourlyStockData[index - i].close
            prev_price = self.hourlyStockData[index - i - 1].close
            ret = (curr_price - prev_price) / (prev_price if prev_price != 0 else 1e-8)
            lagged.append(ret)
        return np.array(lagged, dtype=np.float32)

    def _get_future_returns(self, index: int) -> np.ndarray:
        """
        Return an array of shape (n_future,) with
          ret(t+1), ret(t+2), ..., ret(t+n_future).
        Using index as the "current" step.
        """
        fut = []
        for i in range(1, self.n_future + 1):
            nxt = index + i
            if nxt < len(self.hourlyStockData):
                now_price = self.hourlyStockData[nxt].close
                prev_price = self.hourlyStockData[nxt - 1].close
                fut_ret = (now_price - prev_price) / (
                    prev_price if prev_price != 0 else 1e-8
                )
            else:
                fut_ret = 0.0
            fut.append(fut_ret)
        return np.array(fut, dtype=np.float32)

    def _get_current_indicators(self, index: int) -> np.ndarray:
        """
        Return the current step's technical indicators, if any.
        e.g., macd, ema, sma, rsi
        """
        hv = self.hourlyStockData[index]

        if hv.macd is None:
            macd = 0.0
        else:
            macd = hv.macd

        if hv.ema is None:
            ema = 0.0
        else:
            ema = hv.ema

        if hv.sma is None:
            sma = 0.0
        else:
            sma = hv.sma

        if hv.rsi is None:
            rsi = 50.0
        else:
            rsi = hv.rsi

        return np.array([macd, ema, sma, rsi], dtype=np.float32)

    def _compute_return(self, index: int) -> float:
        """
        Return the 'immediate' return from (index-1) to index, normalized by price at index.
        """
        if index < 1 or index >= len(self.hourlyStockData):
            return 0.0
        price_now = self.hourlyStockData[index].close
        price_prev = self.hourlyStockData[index - 1].close
        if abs(price_now) < 1e-8:
            return 0.0
        return (price_now - price_prev) / price_now

    def _build_observation(self, index: int) -> np.ndarray:
        """
        Build the final observation array:
          [ lagged returns (n_lags) + future returns (n_future) + indicators (if included) ]
        """
        lagged = self._get_lagged_returns(index)
        future = (
            self._get_future_returns(index)
            if self.n_future > 0
            else np.array([], dtype=np.float32)
        )
        if self.include_indicators:
            indicators = self._get_current_indicators(index)
        else:
            indicators = np.array([], dtype=np.float32)

        # Combine everything
        obs = np.concatenate([lagged, future, indicators]).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        super().reset(seed=seed)

        self.hourlyStockData = HourlyStockData(
            self.ticker, self.start_date, self.end_date
        )
        self._episode_ended = False
        self.action = 0
        self.current_step = self.n_lags

        obs = self._build_observation(self.current_step)
        if return_info:
            return obs, {}
        else:
            return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._episode_ended:
            raise RuntimeError("Episode is done. Call env.reset() before step().")

        ret = self._compute_return(self.current_step)

        change_in_position = abs(self.action - action)
        cost = change_in_position * self.tc

        reward = action * ret - cost

        done = False
        truncated = False
        if self.current_step >= len(self.hourlyStockData) - 2:
            done = True
            self._episode_ended = True

        self.current_step += 1
        self.action = action

        obs = self._build_observation(self.current_step)

        current_data = (
            self.hourlyStockData[self.current_step]
            if self.current_step < len(self.hourlyStockData)
            else None
        )
        info = {
            "current_step": self.current_step,
            "date": current_data.timestamp if current_data else None,
            "return": ret,
        }

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.current_step}, Action: {self.action}")
