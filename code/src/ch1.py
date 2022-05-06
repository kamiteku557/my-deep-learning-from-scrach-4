from typing import cast
import numpy as np


def avg_method_1(seed: int = 0):
    rng = np.random.default_rng(seed=seed)
    rewards = []

    for n in range(1, 11):
        reward = rng.random()
        rewards.append(reward)
        Q = sum(rewards) / n
        print(Q)


def avg_method_2(seed: int = 0):
    rng = np.random.default_rng(seed=seed)
    Q = 0
    for n in range(1, 11):
        reward = rng.random()
        Q += (reward - Q) / n
        print(Q)


class Bandit:
    """simple bandit"""

    def __init__(self, arms: int = 10, seed: int = 0):
        self.rng = np.random.default_rng(seed=seed)
        self.rates = self.rng.random(arms)

    def play(self, arm: int) -> int:
        rate = self.rates[arm]
        return int(rate > self.rng.random())


class Agent:
    """epsilon-greedy agent"""

    def __init__(self, epsilon: float, action_size: int = 10, seed: int = 0):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
        self.rng = np.random.default_rng(seed)

    def update(self, action: int, reward: float):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, len(self.Qs))
        return cast(int, np.argmax(self.Qs))


class NonStatBandit:
    """non stational bandit"""

    def __init__(self, arms: int = 10, seed: int = 0):
        self.arms = arms
        self.rng = np.random.default_rng(seed=seed)
        self.rates = self.rng.random(arms)

    def play(self, arm: int) -> int:
        rate = self.rates[arm]
        self.rates += 0.1 * self.rng.standard_normal(self.arms)
        return int(rate > self.rng.random())


class AlphaAgent:
    """update by exponential weighted avg"""

    def __init__(
        self, epsilon: float, alpha: float, action_size: int = 10, seed: int = 0
    ):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def update(self, action: int, reward: float):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, len(self.Qs))
        return cast(int, np.argmax(self.Qs))
