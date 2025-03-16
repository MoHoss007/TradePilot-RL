import yfinance as yf
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from technical_indicators import IndicatorMixin, PivotPoints


@dataclass
class HourlyStockValue:
    timestamp: datetime
    close: float
    high: float
    low: float
    macd: Optional[float] = None
    ema: Optional[float] = None
    sma: Optional[float] = None
    rsi: Optional[float] = None
    pivots: Optional[PivotPoints] = None


class HourlyStockData(IndicatorMixin):
    """
    An iterable class that fetches hourly stock data from Yahoo Finance
    and contains HourlyStockValue objects for each hour in the requested period.
    """

    def __init__(self, ticker: str, start: str, end: str) -> None:
        """
        Initializes the class by downloading hourly data for the specified ticker
        from the start date to the end date.

        :param ticker: The stock ticker symbol (e.g., 'AAPL').
        :param start: The start date in 'YYYY-MM-DD' format.
        :param end: The end date in 'YYYY-MM-DD' format.
        """
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = "1h"
        self._index: int = 0

        df = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            progress=False,
        )

        self.data: List[HourlyStockValue] = []

        if df is None or df.empty:
            return

        for timestamp, row in df.iterrows():
            self.data.append(
                HourlyStockValue(
                    timestamp=timestamp.to_pydatetime(),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                )
            )

    def __iter__(self):
        return self

    def __next__(self) -> HourlyStockValue:
        if self._index >= len(self.data):
            raise StopIteration

        value: HourlyStockValue = self.data[self._index]
        self._index += 1
        return value
