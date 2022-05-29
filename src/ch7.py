from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

HEIGHT = 3
WIDTH = 4


def one_hot(state: Tuple[int, int]) -> TensorType[1, "HEIGHT * WIDTH"]:
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return torch.tensor(vec[np.newaxis, :])


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(HEIGHT * WIDTH, 100)
        self.l2 = nn.Linear(100, 4)

    def forward(self, x: TensorType["batch", "HEIGHT * WIDTH"]):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self, seed: int = 0):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1  # eps-greedy
        self.action_size = 4
        # numpyのseed
        self.rng = np.random.default_rng(seed=seed)

        self.qnet = QNet()
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: Tuple[int, int]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_size)
        qs = self.qnet(state)
        return qs.argmax().item()

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> TensorType[1]:
        if done:
            next_Q = torch.zeros(1)
        else:
            next_Qs = self.qnet(next_state)
            next_Q = next_Qs.max(axis=1)[0]
            next_Q.detach()

        target = self.gamma * next_Q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data
