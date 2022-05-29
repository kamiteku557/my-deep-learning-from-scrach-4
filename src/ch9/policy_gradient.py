from dataclasses import dataclass
from typing import List, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchtyping import TensorType


@dataclass
class RecordOfMemory:
    __slots__ = ("reward", "prob")
    reward: float
    prob: TensorType


class Policy(nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x: TensorType["batch", 4]) -> TensorType["batch", "action_size"]:
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    def __init__(self, update_by_REINFORCE: bool = False):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory: List[RecordOfMemory] = []
        self.pi = Policy(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

        self.update_by_REINFORCE = update_by_REINFORCE

    def get_action(self, state: npt.NDArray) -> Tuple[int, TensorType["batch"]]:
        state = torch.from_numpy(state[np.newaxis, :])  # type: ignore
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = cast(int, m.sample().item())
        return action, probs[action]

    def add(self, reward: float, prob: TensorType):
        data = RecordOfMemory(reward, prob)
        self.memory.append(data)

    def update(self):
        if len(self.memory) == 0:
            raise RuntimeError("Any record doesn't exist in memory.")

        G, loss = 0, torch.tensor(0, dtype=torch.float64)

        if self.update_by_REINFORCE:
            for d in reversed(self.memory):
                G = d.reward + self.gamma * G
                loss += -torch.log(d.prob) * G
        else:
            for d in reversed(self.memory):
                G = d.reward + self.gamma * G
            for d in self.memory:
                loss += -torch.log(d.prob) * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
