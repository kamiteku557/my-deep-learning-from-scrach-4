from typing import List, Tuple
from collections import defaultdict

import numpy as np

from ch5 import RecordOfMemory, greedy_probs


class McOffPolicyAgent:
    """モンテカルロ法を使用して方策制御を行うエージェント(Off Policy)"""

    def __init__(self, seed=0):
        self.gamma = 0.9
        self.action_size = 4
        self.epsilon = 0.1  # eps-greedy
        self.alpha = 0.1  # exponential moving avg
        self.rng = np.random.default_rng(seed=seed)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

        self.pi = defaultdict(lambda: random_actions.copy())
        self.b = defaultdict(lambda: random_actions.copy())
        self.Q = defaultdict(float)
        self.memory: List[RecordOfMemory] = []

    def get_action(self, state: Tuple[int, int]) -> int:
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return self.rng.choice(actions, p=probs)

    def add(self, state: Tuple[int, int], action: int, reward: float):
        self.memory.append(RecordOfMemory(state, action, reward))

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        rho = 1

        for d in reversed(self.memory):
            state, action, reward = d.state, d.action, d.reward
            key = (state, action)

            # update Q function using sample data
            G = self.gamma * rho * G + reward
            self.Q[key] += (G - self.Q[key]) * self.alpha

            rho *= self.pi[state][action] / self.b[state][action]
            self.pi[state] = greedy_probs(self.Q, state, eps=0)
            self.b[state] = greedy_probs(self.Q, state, eps=self.epsilon)
