import pandas as pd
from dataclasses import dataclass


@dataclass
class PivotPoints:
    pivot: float
    r1: float
    s1: float
    r2: float
    s2: float


class IndicatorMixin:
    """
    A mixin that provides methods for calculating technical indicators.
    Assumes the main class has a 'data' attribute which is
    a list of HourlyStockValue objects.
    """

    def _get_close_series(self) -> pd.Series:
        if not hasattr(self, "data"):
            raise AttributeError(
                "The class using IndicatorMixin must have a 'data' attribute "
                "containing a list of HourlyStockValue objects."
            )

        return pd.Series(
            data=[hv.close for hv in self.data],
            index=[hv.timestamp for hv in self.data],
            name="close",
        )

    def _get_dataframe(self) -> pd.DataFrame:
        if not hasattr(self, "data"):
            raise AttributeError(
                "The class using IndicatorMixin must have a 'data' attribute "
                "containing a list of HourlyStockValue objects."
            )

        df = pd.DataFrame(
            {
                "close": [hv.close for hv in self.data],
                "high": [hv.high for hv in self.data],
                "low": [hv.low for hv in self.data],
            },
            index=[hv.timestamp for hv in self.data],
        )
        return df

    def calculate_sma(self, period: int = 20) -> None:
        """
        Calculate Simple Moving Average (SMA) over a given period.

        :param period: The number of periods (hours) to average.
        """
        close_series = self._get_close_series()
        sma = close_series.rolling(window=period).mean()

        for i, hv in enumerate(self.data):
            hv.sma = sma.iloc[i]

    def calculate_ema(self, period: int = 20) -> None:
        """
        Calculate Exponential Moving Average (EMA) over a given period.

        :param period: The number of periods (hours) to average.
        """
        close_series = self._get_close_series()
        ema = close_series.ewm(span=period, adjust=False).mean()

        for i, hv in enumerate(self.data):
            hv.ema = ema.iloc[i]

    def calculate_pivot_points(self) -> None:
        """
        Calculate standard pivot points for each hour data.
        """
        df = self._get_dataframe()

        # Calculate Pivot
        df["Pivot"] = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate R1 & S1
        df["R1"] = 2 * df["Pivot"] - df["low"]
        df["S1"] = 2 * df["Pivot"] - df["high"]

        # Calculate R2 & S2
        df["R2"] = df["Pivot"] + (df["R1"] - df["S1"])
        df["S2"] = df["Pivot"] - (df["R1"] - df["S1"])

        for i, hv in enumerate(self.data):
            hv.pivots = PivotPoints(
                pivot=float(df["Pivot"].iloc[i]),
                r1=float(df["R1"].iloc[i]),
                s1=float(df["S1"].iloc[i]),
                r2=float(df["R2"].iloc[i]),
                s2=float(df["S2"].iloc[i]),
            )

    def calculate_rsi(self, period: int = 14) -> None:
        """
        Calculate the Relative Strength Index (RSI) for the given period.

        :param period: The RSI look-back period (default 14).
        """
        close_series = self._get_close_series()

        delta = close_series.diff(1)

        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        ema_up = up.ewm(com=period - 1, adjust=False).mean()
        ema_down = down.ewm(com=period - 1, adjust=False).mean()

        rs = ema_up / ema_down

        rsi = 100 - (100 / (1 + rs))

        for i, hv in enumerate(self.data):
            hv.rsi = rsi.iloc[i]

    def calculate_macd(
        self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9
    ) -> None:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        :param fastperiod: Fast EMA period (default 12)
        :param slowperiod: Slow EMA period (default 26)
        :param signalperiod: Signal EMA period (default 9)
        """
        close_series = self._get_close_series()

        # Fast and slow exponential moving averages
        ema_fast = close_series.ewm(span=fastperiod, adjust=False).mean()
        ema_slow = close_series.ewm(span=slowperiod, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        # signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
        # histogram = macd_line - signal_line

        for i, hv in enumerate(self.data):
            hv.macd = macd_line.iloc[i]
