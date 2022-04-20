import numpy as np


def avg_method_1(seed=0):
    np.random.seed(seed)
    rewards = []

    for n in range(1, 11):
        reward = np.random.rand()
        rewards.append(reward)
        Q = sum(rewards) / n
        print(Q)


def avg_method_2(seed=0):
    np.random.seed(seed)
    Q = 0
    for n in range(1, 11):
        reward = np.random.rand()
        Q += (reward - Q) / n
        print(Q)


class Bandit:
    """simple bandit"""

    def __init__(self, arms: int = 10) -> None:
        self.rates = np.random.rand(arms)

    def play(self, arm) -> int:
        rate = self.rates[arm]
        return int(rate > np.random.rand())


class Agent:
    """epsilon-greedy agent"""

    def __init__(self, epsilon: float, action_size: int = 10) -> None:
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action: int, reward: float) -> None:
        self.ns[action] += 1
        self.Qs += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
