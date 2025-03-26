from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, state: Union[np.ndarray, Tuple]) -> int:
        """
        Given a state, select an action.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the agent's parameters.
        """
        pass
