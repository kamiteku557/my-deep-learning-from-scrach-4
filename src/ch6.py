from typing import Deque, Tuple
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from ch5 import greedy_probs


class TdAgent:
    """TD法を使用して方策評価を行うエージェント"""

    def __init__(self, seed: int = 0):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4
        self.rng = np.random.default_rng(seed=seed)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions.copy())
        self.V = defaultdict(float)

    def get_action(self, state: Tuple[int, int]) -> int:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return self.rng.choice(actions, p=probs)

    def eval(
        self,
        state: Tuple[int, int],
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V

        # formula 6.9
        self.V[state] += (target - self.V[state]) * self.alpha


@dataclass
class RecordOfMemory:
    __slots__ = ("state", "action", "reward", "done")
    state: Tuple[int, int]
    action: int
    reward: float
    done: bool


class SarsaAgent:
    """SARSA"""

    def __init__(self, seed: int = 0):
        self.gamma = 0.9
        self.action_size = 4
        self.epsilon = 0.1  # eps-greedy
        self.alpha = 0.8  # exponential moving avg
        self.rng = np.random.default_rng(seed=seed)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(float)
        self.memory: Deque[RecordOfMemory] = deque(maxlen=2)  # t, t+1のデータのみ保持

    def get_action(self, state: Tuple[int, int]) -> int:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return self.rng.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        done: bool,
    ):
        self.memory.append(RecordOfMemory(state, action, reward, done))
        if len(self.memory) < 2:
            return

        # data of t
        d = self.memory[0]
        # data of t + 1
        d_next = self.memory[1]

        next_Q = 0 if d.done else self.Q[d_next.state, d_next.action]

        target = d.reward + self.gamma * next_Q
        self.Q[d.state, d.action] += (target - self.Q[d.state, d.action]) * self.alpha
        self.pi[d.state] = greedy_probs(self.Q, d.state, self.epsilon)


class SarsaOffPolicyAgent:
    """方策OFF型のSARSA"""

    def __init__(self, seed: int = 0):
        self.gamma = 0.9
        self.action_size = 4
        self.epsilon = 0.1  # eps-greedy
        self.alpha = 0.8  # exponential moving avg
        self.rng = np.random.default_rng(seed=seed)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        # target policy
        self.pi = defaultdict(lambda: random_actions)
        # behavior policy
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(float)
        self.memory: Deque[RecordOfMemory] = deque(maxlen=2)  # t, t+1のデータのみ保持

    def get_action(self, state: Tuple[int, int]):
        # get action by behavior policy
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return self.rng.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        done: bool,
    ):
        self.memory.append(RecordOfMemory(state, action, reward, done))
        if len(self.memory) < 2:
            return

        # data of t
        d = self.memory[0]
        # data of t + 1
        d_next = self.memory[1]

        # importance sampling
        if d.done:
            next_Q = 0
            rho = 1
        else:
            next_Q = self.Q[d_next.state, d_next.action]
            rho = (
                self.pi[d_next.state][d_next.action]
                / self.b[d_next.state][d_next.action]
            )

        target = rho * (d.reward + self.gamma * next_Q)
        self.Q[d.state, d.action] += (target - self.Q[d.state, d.action]) * self.alpha

        # update policies
        self.pi[d.state] = greedy_probs(self.Q, d.state, eps=0)
        self.b[d.state] = greedy_probs(self.Q, d.state, eps=self.epsilon)


class QLearningAgent:
    """Q学習を行うエージェント"""

    def __init__(self, seed: int = 0):
        self.gamma = 0.9
        self.action_size = 4
        self.epsilon = 0.1  # eps-greedy
        self.alpha = 0.8  # exponential moving avg
        self.rng = np.random.default_rng(seed=seed)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        # target policy
        self.pi = defaultdict(lambda: random_actions)
        # behavior policy
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(float)

    def get_action(self, state: Tuple[int, int]):
        # get action by behavior policy
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return self.rng.choice(actions, p=probs)

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ):
        if done:
            next_Q_max = 0
        else:
            get_q = lambda a: self.Q[next_state, a]  # noqa
            next_Q_max = max(map(get_q, range(self.action_size)))

        target = reward + self.gamma * next_Q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # update policies
        self.pi[state] = greedy_probs(self.Q, state, eps=0)
        self.b[state] = greedy_probs(self.Q, state, eps=self.epsilon)
