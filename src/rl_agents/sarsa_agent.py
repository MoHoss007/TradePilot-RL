import numpy as np
from typing import Tuple, Union
from collections import defaultdict
from src.rl_agents.base_agent import BaseAgent
import pickle


class SarsaAgent(BaseAgent):
    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        """
        :param n_actions: Number of discrete actions (here: 0=hold, 1=buy, 2=sell).
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Exploration rate for epsilon-greedy policy.
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = defaultdict(self._default_q_values)

        self.inventory = []

    def _default_q_values(self) -> np.ndarray:
        return np.zeros(self.n_actions, dtype=float)

    def get_action(self, state: Union[np.ndarray, Tuple[int, ...]]) -> int:
        """
        Epsilon-greedy action selection given a discrete state.
        Converts state to a tuple if needed.
        """
        if isinstance(state, np.ndarray):
            state = tuple(state[0].astype(int))

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(self.Q[state]))

        # Prevent sell action if inventory is empty.
        if action == 2 and len(self.inventory) == 0:
            action = 0

        return action

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        next_action: int,
        done: bool,
    ):
        """
        SARSA update:
          Q(s,a) += alpha * [ r + gamma * Q(s',a') - Q(s,a) ]
        If done, then Q(s',a') is treated as 0.
        """
        predict = self.Q[state][action]
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] = predict + self.alpha * (target - predict)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)

    def save_weights(self, filename: str = "sarsa_qtable.pkl"):
        """
        Save the Q-table ('weights') to a .pkl file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.Q, f)
        print(f"Q-table saved to {filename}")

    def load_wights(self, filename: str = "sarsa_qtable.pkl"):
        """
        Load the Q-table ('weights') from a .pkl file.
        """
        with open(filename, "rb") as f:
            self.Q = pickle.load(f)
