import yfinance as yf
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from src.data_loader.technical_indicators import IndicatorMixin, PivotPoints


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

        start_date = datetime.strptime(self.start, "%Y-%m-%d")
        end_date = datetime.strptime(self.end, "%Y-%m-%d")
        if (datetime.now() - start_date).days > 730 or (
            datetime.now() - end_date
        ).days > 730:
            raise ValueError(
                "The requested date range must be within the last 730 days."
            )

        df = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            progress=False,
        )

        self._data: List[HourlyStockValue] = []

        if df is None or df.empty:
            return

        for timestamp, row in df.iterrows():
            self._data.append(
                HourlyStockValue(
                    timestamp=timestamp.to_pydatetime(),  # type: ignore
                    close=float(row.iloc[0]),  # Close price
                    high=float(row.iloc[1]),  # High price
                    low=float(row.iloc[2]),  # Low price
                )
            )

    def __iter__(self):
        return self

    def __next__(self) -> HourlyStockValue:
        if self._index >= len(self._data):
            raise StopIteration

        value: HourlyStockValue = self._data[self._index]
        self._index += 1
        return value

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)
