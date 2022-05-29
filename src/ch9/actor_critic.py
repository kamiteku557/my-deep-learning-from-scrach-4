from typing import Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchtyping import TensorType


class PolicyNet(nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x: TensorType["batch", 4]) -> TensorType["batch", "action_size"]:
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x: TensorType["batch", 4]) -> TensorType["batch", 1]:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state: npt.NDArray) -> Tuple[int, TensorType["batch"]]:
        state = torch.from_numpy(state[np.newaxis, :])  # type: ignore
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = cast(int, m.sample().item())
        return action, probs[action]

    def update(
        self,
        state: npt.NDArray,
        action_prob,
        reward: float,
        next_state: npt.NDArray,
        done: int,
    ):
        state = torch.from_numpy(state[np.newaxis, :])  # type: ignore
        next_state = torch.from_numpy(next_state[np.newaxis, :])  # type: ignore

        target: TensorType = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()

        v: TensorType = self.v(state)
        loss_v: TensorType = nn.MSELoss()(v, target)

        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()

        self.optimizer_v.step()
        self.optimizer_pi.step()
