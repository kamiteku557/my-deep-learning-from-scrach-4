from typing import List, Tuple, TypeVar
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

T = TypeVar("T")


@dataclass
class RecordOfMemory:
    __slots__ = ("state", "action", "reward")
    state: Tuple[int, int]
    action: int
    reward: float


class RandomAgent:
    """モンテカルロ法を使用して方策評価を行うエージェント"""

    def __init__(self, seed: int = 0):
        self.gamma = 0.9
        self.action_size = 4
        self.rng = np.random.default_rng(seed=seed)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions.copy())
        self.V = defaultdict(float)
        self.cnts = defaultdict(float)
        self.memory: List[RecordOfMemory] = []

    def get_action(self, state: Tuple[int, int]) -> int:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return self.rng.choice(actions, p=probs)

    def add(self, state: Tuple[int, int], action: int, reward: float):
        self.memory.append(RecordOfMemory(state, action, reward))

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for d in reversed(self.memory):
            reward, state = d.reward, d.state
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


class McAgent:
    """モンテカルロ法を使用して方策制御を行うエージェント"""

    def __init__(self, seed=0):
        self.gamma = 0.9
        self.action_size = 4
        self.epsilon = 0.1  # eps-greedy
        self.alpha = 0.1  # exponential moving avg
        self.rng = np.random.default_rng(seed=seed)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(float)
        self.memory: List[RecordOfMemory] = []

    def get_action(self, state: Tuple[int, int]) -> int:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return self.rng.choice(actions, p=probs)

    def add(self, state: Tuple[int, int], action: int, reward: float):
        self.memory.append(RecordOfMemory(state, action, reward))

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for d in reversed(self.memory):
            state, action, reward = d.state, d.action, d.reward
            G = self.gamma * G + reward
            key = (state, action)

            self.Q[key] += (G - self.Q[key]) * self.alpha

            self.pi[state] = greedy_probs(self.Q, state, eps=self.epsilon)


def greedy_probs(
    Q: dict[tuple[T, int], float], state: T, eps: float = 0.0, action_size: int = 4
) -> dict[int, float]:
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = eps / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1 - eps

    return action_probs
